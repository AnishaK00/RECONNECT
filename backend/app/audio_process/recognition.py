import os
import json
import uuid
import shutil
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.spatial.distance import cosine
from scipy.io import wavfile
import torchaudio

from database.db import get_all_identities, voice_blob_to_embeddings

# Patch torchaudio for speechbrain compatibility with newer torchaudio versions
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

if not hasattr(torchaudio, 'io'):
    class DummyIO:
        class StreamReader:
            pass
    torchaudio.io = DummyIO()

import huggingface_hub
_original_hf_download = huggingface_hub.hf_hub_download

def _patched_hf_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    try:
        return _original_hf_download(*args, **kwargs)
    except huggingface_hub.utils.EntryNotFoundError as e:
        filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
        if filename == "custom.py":
            # Speechbrain looks for a custom.py that doesn't exist for this model.
            # Return an empty dummy file to satisfy it.
            dummy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy_custom.py")
            if not os.path.exists(dummy_path):
                open(dummy_path, 'w').close()
            return dummy_path
        raise e

huggingface_hub.hf_hub_download = _patched_hf_download

import torch

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    # Handle older speechbrain versions if necessary
    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        EncoderClassifier = None


def load_audio_tensor(audio_path):
    """Load a PCM WAV without routing through TorchCodec."""
    sample_rate, audio = wavfile.read(audio_path)
    if audio.dtype.kind in "iu":
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio).unsqueeze(0), int(sample_rate)


class AudioEmbedder:
    """Handles ECAPA-TDNN embedding extraction."""
    
    def __init__(self, savedir=None):
        print("Initializing AudioEmbedder (SpeechBrain)...")
        if EncoderClassifier is None:
            raise ImportError("speechbrain is not installed. Run: pip install speechbrain")

        backend_root = Path(__file__).resolve().parents[2]
        savedir = Path(savedir) if savedir else backend_root / "tmp_speechbrain"
        savedir = savedir.resolve()
        hyperparams_path = savedir / "hyperparams.yaml"

        # On Windows, a cache copied from Linux can contain the symlink target
        # as plain text instead of the YAML file SpeechBrain expects.
        if hyperparams_path.is_file():
            try:
                hyperparams_text = hyperparams_path.read_text(encoding="utf-8")
            except OSError:
                hyperparams_text = ""

            if "modules:" not in hyperparams_text or "pretrainer:" not in hyperparams_text:
                hyperparams_path.unlink()

        local_model_dir = backend_root.parent / "pretrained_models" / "spkrec-ecapa-voxceleb"
        if not local_model_dir.is_dir():
            raise FileNotFoundError(
                f"Local SpeechBrain model directory not found: {local_model_dir}"
            )

        savedir.mkdir(parents=True, exist_ok=True)
        source_yaml = local_model_dir / "hyperparams.yaml"
        local_yaml = source_yaml.read_text(encoding="utf-8")
        local_yaml = local_yaml.replace(
            "pretrained_path: speechbrain/spkrec-ecapa-voxceleb",
            f"pretrained_path: {local_model_dir.as_posix()}"
        ).replace(
            "label_encoder.txt",
            "label_encoder.ckpt"
        )
        hyperparams_path.write_text(local_yaml, encoding="utf-8")

        for filename in (
            "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt",
            "classifier.ckpt",
            "label_encoder.ckpt",
        ):
            destination = savedir / filename
            if not destination.exists():
                shutil.copy2(local_model_dir / filename, destination)

        (savedir / "custom.py").touch(exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.embedding_model = EncoderClassifier.from_hparams(
            source=str(savedir),
            savedir=str(savedir),
            run_opts={"device": self.device}
        )

    def extract_embedding(self, audio_path):
        """Extracts ECAPA-TDNN embedding from a WAV file."""
        fs, audio = wavfile.read(audio_path)
        if audio.dtype.kind in "iu":
            max_value = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_value
        else:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        signal = torch.from_numpy(audio).unsqueeze(0)
        # SpeechBrain models typically expect 16kHz
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        # Ensure mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        signal = signal.to(self.device)
        embeddings = self.embedding_model.encode_batch(signal)
        # shape: (1, 1, 192) usually, convert to 1D numpy array
        return embeddings.squeeze().cpu().numpy()


class AudioMatcher:
    """Matches live embeddings against enrolled SQLite voice embeddings."""
    
    def __init__(self, gallery_path="data/embeddings/", threshold=0.65):
        """Initialize AudioMatcher with a matching threshold.
        Updated default threshold to 0.65 for both enrolled and unenrolled identity matching.
        """
        self.gallery_path = gallery_path
        self.threshold = threshold
        self.known_embeddings = self._load_gallery()

    def _load_gallery(self):
        """Load all enrolled voice templates from SQLite."""
        known = {}

        for row in get_all_identities():
            if row["voice_embedding"] is not None:
                for embedding in voice_blob_to_embeddings(row["voice_embedding"]):
                    if embedding.ndim == 1 and np.isfinite(embedding).all():
                        known.setdefault(row["name"], []).append(embedding)

        return known

    def refresh(self):
        """Refresh SQLite templates so enrollment changes need no restart."""
        self.known_embeddings = self._load_gallery()

    def match(self, embedding):
        """Compare against every enrolled SQLite voice template."""
        self.refresh()
        best_match, best_score = self.best_match(embedding)
        if best_match is not None and best_score >= self.threshold:
            return best_match, best_score
        return None, best_score

    def best_match(self, embedding):
        """Return the highest cosine similarity without applying the threshold."""

        best_match = None
        best_score = -1.0
        candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
        candidate_norm = np.linalg.norm(candidate)

        if candidate_norm == 0 or not np.isfinite(candidate).all():
            return None, best_score

        candidate = candidate / candidate_norm

        for person_name, known_embs in self.known_embeddings.items():
            for known_emb in known_embs:
                reference = np.asarray(known_emb, dtype=np.float32).reshape(-1)
                if reference.shape != candidate.shape:
                    continue
                reference_norm = np.linalg.norm(reference)
                if reference_norm == 0 or not np.isfinite(reference).all():
                    continue
                sim = float(np.dot(candidate, reference / reference_norm))
                if sim > best_score:
                    best_score = sim
                    best_match = person_name

        return best_match, best_score


class SessionSpeakerMemory:
    """
    Remember speakers already identified inside one live stream session.

    Recognition still runs per recording against the gallery. When a later
    recording of the same person dips just below the gallery threshold, reuse
    the session's earlier confident match instead of creating a new unknown.
    """

    def __init__(
        self,
        continuity_threshold: float = 0.58,
        session_embed_threshold: float = 0.60,
    ):
        self.continuity_threshold = continuity_threshold
        self.session_embed_threshold = session_embed_threshold
        self._lock = Lock()
        # session_id -> identity -> embeddings observed this session
        self._embeddings: dict[str, dict[str, list[np.ndarray]]] = {}
        self._confirmed: dict[str, set[str]] = {}

    def remember(
        self,
        session_id: str,
        identity: str,
        embedding: np.ndarray,
        *,
        confirmed: bool = True,
    ) -> None:
        if not session_id or not identity or embedding is None:
            return
        with self._lock:
            by_identity = self._embeddings.setdefault(session_id, {})
            by_identity.setdefault(identity, []).append(
                np.asarray(embedding, dtype=np.float32).reshape(-1)
            )
            if confirmed:
                self._confirmed.setdefault(session_id, set()).add(identity)

    def resolve(
        self,
        session_id: str,
        embedding: np.ndarray,
        gallery_best_id: str | None,
        gallery_best_score: float,
    ) -> tuple[str | None, float, str | None]:
        """
        Try to keep the same identity across recordings in one session.

        Returns (identity, score, reason) or (None, score, None).
        """

        flat = np.asarray(embedding, dtype=np.float32).reshape(-1)

        # Near-miss against someone already confirmed earlier in this session.
        if (
            gallery_best_id is not None
            and gallery_best_score >= self.continuity_threshold
            and gallery_best_id in self._confirmed.get(session_id, set())
        ):
            return gallery_best_id, gallery_best_score, "session_continuity"

        with self._lock:
            session_embs = self._embeddings.get(session_id, {})

        best_id = None
        best_score = -1.0
        for identity, embs in session_embs.items():
            for known_emb in embs:
                sim = 1 - cosine(flat, known_emb.flatten())
                if sim > best_score:
                    best_score = float(sim)
                    best_id = identity

        if best_id is not None and best_score >= self.session_embed_threshold:
            return best_id, best_score, "session_embedding"

        return None, gallery_best_score, None




class AudioAggregator:
    """Collects segments over a short time window to prevent ghost profiles.

    The default ``min_observations`` is taken from ``config.MIN_UNKNOWN_OBSERVATIONS``
    so that the threshold can be adjusted centrally.
    """
    
    def __init__(self, min_observations=None):
        # If not provided, use the configured minimum observations for unknowns.
        from config import MIN_UNKNOWN_OBSERVATIONS
        self.min_observations = min_observations if min_observations is not None else MIN_UNKNOWN_OBSERVATIONS
        self.observation_buffer = {}  # speaker_id -> count

    def add_observation(self, speaker_id):
        if speaker_id not in self.observation_buffer:
            self.observation_buffer[speaker_id] = 0
        self.observation_buffer[speaker_id] += 1
        return self.observation_buffer[speaker_id] >= self.min_observations
        
    def reset(self):
        self.observation_buffer.clear()
