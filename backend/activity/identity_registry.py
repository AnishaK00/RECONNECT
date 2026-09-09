"""
identity_registry.py — Cross-modal person identity resolution for RECONNECT
=============================================================================

Drop this file next to active_speaker_detection.py and import PersonRegistry
into it. It replaces the ad-hoc "Unknown_001" counters in FaceDetector /
SpeakerRecogniser and the confirmation logic in SpeakerFaceMatcher with a
single shared registry so all three modalities agree on one set of IDs.

What it solves
---------------
1. Unenrolled face/voice IDs now persist across the WHOLE video instead of
   resetting every frame/turn.
2. Voice enrolled + face not yet enrolled: the first time that voice is
   heard while a face is confidently lip-synced to it, the face embedding
   gets attached to that same known person. Next time that face alone
   appears, it already resolves to the known name.
3. Face enrolled + voice not yet enrolled: symmetric — first confident
   lip-synced pairing teaches the registry that voice.
4. Both unenrolled: the first confident lip-synced pairing merges the two
   session clusters into one Unknown_N person instead of leaving two
   disconnected labels.
5. Both missing (no face detected / no usable audio for a turn): resolved
   as UNCONFIRMED without inventing a link.
6. Off-screen voice: if TalkNet's best on-screen face match has low
   confidence (below ASD_LINK_THRESHOLD), the voice is NOT force-linked to
   whichever face happens to be visible. The voice is tracked as its own
   person ("heard, not seen this turn") and any visible-but-not-speaking
   face is tracked as ITS own person ("seen, not speaking this turn").
   This covers both "unknown bg voice + known/unknown face on screen" and
   "known bg voice + unknown/known face on screen".
7. Confidence gating: a person only becomes "confirmed" (worth surfacing
   as a real recurring participant) once they've been the ASD-confirmed
   speaker in MIN_SPEAKING_TURNS_FOR_CONFIRMED turns, OR they're enrolled.
   Below that, they still get a stable label for visualization but are
   flagged LINKED_UNCONFIRMED instead of DUAL_CONFIRMED — so you don't have
   to enroll every background voice, but a real repeated speaker still
   accumulates evidence and eventually locks in.

Naming happens later
---------------------
Nothing here requires a human name. person_id (e.g. "Unknown_003") is what
gets shown during the session; PersonRecord.name stays None until a
caregiver/family member assigns one (call registry.assign_name(person_id,
"Mom") whenever that happens — see bottom of file).
"""

from dataclasses import dataclass, field
import numpy as np
import torch
from scipy.spatial.distance import cosine

FACE_MATCH_THRESHOLD              = 0.50   # slightly relaxed — same person across
                                            # lighting/expression can dip below 0.60
VOICE_MATCH_THRESHOLD             = 0.70
ASD_LINK_THRESHOLD                = 0.05
MIN_SPEAKING_TURNS_FOR_CONFIRMED  = 3
MIN_SPEAKER_FACE_OBSERVATIONS = 2

MAX_FACE_SAMPLES = 25  # short clips (a few hundred frames) can just keep every
                       # sample rather than evicting early ones via FIFO


@dataclass
class PersonRecord:
    person_id: str
    name: str = None                       # filled in later by caregiver/family
    face_embedding: np.ndarray = None       # kept for enrollment (single reference photo)
    face_gallery: list = field(default_factory=list)  # session-collected samples, best-match
    voice_embedding: torch.Tensor = None
    face_samples: int = 0
    voice_samples: int = 0
    speaking_turns: int = 0                # confirmed ASD-linked speaking turns
    face_only_sightings: int = 0           # on screen, never confirmed speaking
    voice_only_sightings: int = 0          # heard, never matched to an on-screen face
    is_enrolled_face: bool = False
    is_enrolled_voice: bool = False

    def is_confirmed(self, min_turns: int = 3) -> bool:
        """Enough evidence to treat this as a real recurring participant,
        not a one-off background detection."""
        return self.is_enrolled_face or self.is_enrolled_voice or \
               self.speaking_turns >= min_turns

    def display_label(self) -> str:
        return self.name if self.name else self.person_id


class PersonRegistry:
    """
    One instance per video. Shared by FaceDetector, SpeakerRecogniser and
    SpeakerFaceMatcher (via resolve_turn) so all three agree on person_ids.
    """

    def __init__(self, enrolled_voices: dict = None):
        self.people: dict[str, PersonRecord] = {}
        self._aliases: dict[str, str] = {}   # merged-away id -> surviving id
        self._diarization_label_map: dict[str, str] = {}
        self._turn_results_by_speaker = {}  

        self._speaker_face_evidence = {}

        self.min_speaking_turns_for_confirmed = MIN_SPEAKING_TURNS_FOR_CONFIRMED
        self._next_id = 1

        for name, emb in (enrolled_voices or {}).items():
            rec = self._get_or_create_by_name(name)
            rec.voice_embedding, rec.voice_samples = emb, 1
            rec.is_enrolled_voice = True

    #setup helpers ─────────────────────────────────────────
    def set_total_video_turns(self, total_turns: int):
        if total_turns <= 3:
            self.min_speaking_turns_for_confirmed = 1
        else:
            self.min_speaking_turns_for_confirmed = MIN_SPEAKING_TURNS_FOR_CONFIRMED

    def _get_or_create_by_name(self, name: str) -> PersonRecord:
        for rec in self.people.values():
            if rec.name == name:
                return rec
        rec = PersonRecord(person_id=name, name=name)
        self.people[name] = rec
        return rec

    def ensure_face_identity(self, identity: str) -> PersonRecord:
        """Create the session record for a Who Is This face match."""
        rec = self._get_or_create_by_name(identity)
        rec.is_enrolled_face = True
        return rec

    def _new_id(self) -> str:
        pid = f"Unknown_{self._next_id:03d}"
        self._next_id += 1
        return pid

    @staticmethod
    def _face_sim(a, b) -> float:
        return float(1 - cosine(a, b))

    @staticmethod
    def _voice_sim(a, b) -> float:
        return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    # ── face side (call from FaceDetector instead of an unknown counter) ─
    def identify_face(self, embedding: np.ndarray):
        """
        Returns (person_id, similarity, is_new_person).

        Matches against the enrolled reference embedding (if any) AND the
        best-matching sample in each person's session gallery — not a
        blurred running average — so pose/lighting variation across frames
        doesn't fragment one person into several Unknown_ ids.
        """
        best_id, best_sim = None, 0.0
        for pid, rec in self.people.items():
            candidates = list(rec.face_gallery)
            if rec.face_embedding is not None:
                candidates.append(rec.face_embedding)
            for ref_emb in candidates:
                sim = self._face_sim(embedding, ref_emb)
                if sim > best_sim:
                    best_sim, best_id = sim, pid

        if best_id is not None and best_sim >= FACE_MATCH_THRESHOLD:
            self._update_face(best_id, embedding)
            return best_id, best_sim, False

        pid = self._new_id()
        self.people[pid] = PersonRecord(person_id=pid)
        self._update_face(pid, embedding)
        return pid, 0.0, True

    def peek_face_match(self, embedding: np.ndarray):
        """
        Read-only face lookup — returns (person_id, similarity) for the
        best match, or (None, 0.0) if nothing clears FACE_MATCH_THRESHOLD.
        Unlike identify_face, this never creates a new person or mutates
        any gallery — used for scoring against already-established
        identities without side effects.
        """
        best_id, best_sim = None, 0.0
        for pid, rec in self.people.items():
            candidates = list(rec.face_gallery)
            if rec.face_embedding is not None:
                candidates.append(rec.face_embedding)
            for ref_emb in candidates:
                sim = self._face_sim(embedding, ref_emb)
                if sim > best_sim:
                    best_sim, best_id = sim, pid
        if best_id is not None and best_sim >= FACE_MATCH_THRESHOLD:
            return self.resolve(best_id), best_sim
        return None, 0.0

    def _update_face(self, pid: str, embedding: np.ndarray):
        rec = self.people[pid]
        rec.face_gallery.append(embedding)
        if len(rec.face_gallery) > MAX_FACE_SAMPLES:
            rec.face_gallery.pop(0)   # drop oldest, keep gallery bounded
        rec.face_samples += 1

    def reinforce_face(self, person_id: str, embedding: np.ndarray):
        """
        Add a sample to an ALREADY-KNOWN person's gallery without running
        embedding matching — for when identity came from spatial/temporal
        tracking (frame-to-frame continuity) rather than re-derivation.
        Keeps that person's gallery current for when tracking breaks later
        and embedding matching is needed again.
        """
        pid = self.resolve(person_id)
        if pid in self.people:
            self._update_face(pid, embedding)

    # ── voice side (call from SpeakerRecogniser instead of an unknown counter) ─
    def identify_voice(self, embedding: torch.Tensor):
        best_id, best_sim = None, 0.0
        for pid, rec in self.people.items():
            if rec.voice_embedding is None:
                continue
            sim = self._voice_sim(embedding, rec.voice_embedding)
            if sim > best_sim:
                best_sim, best_id = sim, pid

        if best_id is not None and best_sim >= VOICE_MATCH_THRESHOLD:
            self._update_voice(best_id, embedding)
            return best_id, best_sim, False

        pid = self._new_id()
        self.people[pid] = PersonRecord(person_id=pid)
        self._update_voice(pid, embedding)
        return pid, 0.0, True

    def identify_voice_by_turn(self, speaker_label: str, embedding: torch.Tensor):
        """
        Preferred entry point for per-turn voice recognition.

        PyAnnote's diarization already clusters turns by voice across the
        WHOLE video far more reliably than re-matching short (1-3s) turn
        embeddings against each other with ECAPA — so trust speaker_label
        ('SPEAKER_00', 'SPEAKER_01', ...) for "is this the same voice as
        an earlier turn", and only use ECAPA to check whether that
        diarization cluster matches an ENROLLED voice.

        Returns (person_id, similarity, is_new_person) — same shape as
        identify_voice, so callers don't need to branch on which was used.
        """
        if speaker_label in self._diarization_label_map:
            pid = self.resolve(self._diarization_label_map[speaker_label])
            self._update_voice(pid, embedding)
            rec = self.people[pid]
            sim = self._voice_sim(embedding, rec.voice_embedding) if rec.voice_embedding is not None else 1.0
            return pid, sim, False

        # New diarization cluster for this video — check against enrolled
        # voices only (session Unknown_ clusters are irrelevant here since
        # diarization itself is the session clustering now).
        best_id, best_sim = None, 0.0
        for pid, rec in self.people.items():
            if rec.is_enrolled_voice and rec.voice_embedding is not None:
                sim = self._voice_sim(embedding, rec.voice_embedding)
                if sim > best_sim:
                    best_sim, best_id = sim, pid

        if best_id is not None and best_sim >= VOICE_MATCH_THRESHOLD:
            self._diarization_label_map[speaker_label] = best_id
            self._update_voice(best_id, embedding)
            return best_id, best_sim, False

        pid = self._new_id()
        self.people[pid] = PersonRecord(person_id=pid)
        self._update_voice(pid, embedding)
        self._diarization_label_map[speaker_label] = pid
        return pid, 0.0, True

    def _update_voice(self, pid: str, embedding: torch.Tensor):
        rec = self.people[pid]
        if rec.voice_embedding is None:
            rec.voice_embedding = embedding
        else:
            rec.voice_embedding = (rec.voice_embedding * rec.voice_samples + embedding) / (rec.voice_samples + 1)
        rec.voice_samples += 1

    # ── merging two clusters once evidence says they're the same person ──
    def merge(self, id_a: str, id_b: str) -> str:
        id_a, id_b = self.resolve(id_a), self.resolve(id_b)
        if id_a == id_b:
            return id_a

        rec_a, rec_b = self.people[id_a], self.people[id_b]

        # Whichever side is enrolled / already named survives as the id.
        # If neither is, keep whichever side has more accumulated evidence
        # (a face id with a full Step-2 gallery should survive over a
        # voice id that was just created this turn with one sample).
        def _evidence(rec):
            return rec.face_samples + rec.voice_samples

        if (rec_b.is_enrolled_face or rec_b.is_enrolled_voice or rec_b.name) and \
           not (rec_a.is_enrolled_face or rec_a.is_enrolled_voice or rec_a.name):
            keep_id, drop_id = id_b, id_a
        elif (rec_a.is_enrolled_face or rec_a.is_enrolled_voice or rec_a.name) and \
             not (rec_b.is_enrolled_face or rec_b.is_enrolled_voice or rec_b.name):
            keep_id, drop_id = id_a, id_b
        elif _evidence(rec_b) > _evidence(rec_a):
            keep_id, drop_id = id_b, id_a
        else:
            keep_id, drop_id = id_a, id_b

        keep, drop = self.people[keep_id], self.people[drop_id]

        if keep.face_embedding is None and drop.face_embedding is not None:
            keep.face_embedding = drop.face_embedding
            keep.is_enrolled_face = keep.is_enrolled_face or drop.is_enrolled_face
        keep.face_gallery = (keep.face_gallery + drop.face_gallery)[-MAX_FACE_SAMPLES:]
        keep.face_samples += drop.face_samples
        if keep.voice_embedding is None and drop.voice_embedding is not None:
            keep.voice_embedding, keep.voice_samples = drop.voice_embedding, drop.voice_samples
            keep.is_enrolled_voice = keep.is_enrolled_voice or drop.is_enrolled_voice

        keep.speaking_turns       += drop.speaking_turns
        keep.face_only_sightings  += drop.face_only_sightings
        keep.voice_only_sightings += drop.voice_only_sightings

        del self.people[drop_id]
        self._aliases[drop_id] = keep_id
        for alias, target in list(self._aliases.items()):
            if target == drop_id:
                self._aliases[alias] = keep_id

        return keep_id

    def resolve(self, pid: str) -> str:
        """Follow merge aliases in case an id you're holding got folded
        into another one since you last looked it up."""
        while pid in self._aliases:
            pid = self._aliases[pid]
        return pid

    def record_speaker_face_evidence(
        self,
        speaker_label: str,
        face_id: str,
        asd_score: float
    ):
        """
        Accumulate evidence that a PyAnnote speaker corresponds to
        a persistent face identity.

        IMPORTANT:
        This does NOT immediately merge identities.

        It allows multiple speaking turns to establish the association
        before we call it reliable.
        """
        if not speaker_label or not face_id:
            return

        face_id = self.resolve(face_id)

        if speaker_label not in self._speaker_face_evidence:
            self._speaker_face_evidence[speaker_label] = {}

        speaker_evidence = self._speaker_face_evidence[speaker_label]

        if face_id not in speaker_evidence:
            speaker_evidence[face_id] = {
                "score_sum": 0.0,
                "observations": 0,
                "max_score": 0.0,
            }

        entry = speaker_evidence[face_id]

        entry["score_sum"] += float(asd_score)
        entry["observations"] += 1
        entry["max_score"] = max(
            entry["max_score"],
            float(asd_score)
        )

    def speaker_face_associations(self):
        """
        Return the strongest observed face association for every
        PyAnnote speaker.

        Example:

        {
            "SPEAKER_00": {
                "face_id": "Unknown_001",
                "confidence": 0.81,
                "observations": 4
            },
            "SPEAKER_01": {
                "face_id": "Unknown_002",
                "confidence": 0.77,
                "observations": 3
            }
        }

        The association is based on accumulated TalkNet evidence,
        not a single frame.
        """
        associations = {}

        for speaker_label, faces in self._speaker_face_evidence.items():

            if not faces:
                continue

            ranked = []

            for face_id, evidence in faces.items():

                observations = evidence["observations"]

                if observations <= 0:
                    continue

                avg_score = (
                    evidence["score_sum"] / observations
                )

                ranked.append(
                    (
                        face_id,
                        avg_score,
                        observations,
                        evidence["max_score"],
                    )
                )

            if not ranked:
                continue

            ranked.sort(
                key=lambda x: (
                    x[2],       # observations first
                    x[1],       # average ASD second
                    x[3],       # maximum ASD third
                ),
                reverse=True
            )

            best_face, avg_score, observations, max_score = ranked[0]

            associations[speaker_label] = {
                "face_id": self.resolve(best_face),
                "confidence": round(float(avg_score), 4),
                "observations": observations,
                "max_confidence": round(float(max_score), 4),
            }

        return associations

    def finalize_speaker_face_links(self, min_turns=1, min_avg_confidence=0.08, min_win_ratio=0.6):
        associations = self.speaker_face_associations()
        finalized = {}

        for speaker_label, info in associations.items():
            if speaker_label not in self._diarization_label_map:
                finalized[speaker_label] = None
                continue

            voice_id = self.resolve(self._diarization_label_map[speaker_label])
            face_id = info["face_id"]
            avg_conf = info["confidence"]

            turn_log = self._turn_results_by_speaker.get(speaker_label, [])
            total_turns = len(turn_log)
            wins = sum(1 for t in turn_log if t["top_face_id"] == face_id)
            win_ratio = wins / total_turns if total_turns else 0.0

            print(f"  [finalize check] {speaker_label}: face={face_id} turns={total_turns} "
                f"wins={wins} win_ratio={win_ratio:.2f} avg_conf={avg_conf:.3f}")

            enough_evidence = (
                total_turns >= min_turns
                and avg_conf >= min_avg_confidence
                and win_ratio >= min_win_ratio
            )

            finalized[speaker_label] = self.merge(voice_id, face_id) if enough_evidence else None

        return finalized
    
    def resolve_turn(
        self,
        voice_embedding,
        voice_speaker_label,
        face_candidates: list
    ):
        """
        Resolve one PyAnnote speaking turn.

        face_candidates:
            [
                ("Unknown_001", 0.81),
                ("Unknown_002", 0.32),
            ]

        We record ALL TalkNet candidates as evidence, but only use
        the strongest candidate as the potential active face.
        """

        voice_id = voice_sim = None

        if voice_embedding is not None and voice_speaker_label is not None:

            voice_id, voice_sim, _ = self.identify_voice_by_turn(
                voice_speaker_label,
                voice_embedding
            )

            voice_id = self.resolve(voice_id)

        # ------------------------------------------------------
        # Record PyAnnote speaker -> TalkNet face evidence
        # ------------------------------------------------------

        if voice_speaker_label and face_candidates:

            for face_id, asd_score in face_candidates:

                self.record_speaker_face_evidence(
                    speaker_label=voice_speaker_label,
                    face_id=face_id,
                    asd_score=asd_score
                )

        # ------------------------------------------------------
        # Select strongest visible face
        # ------------------------------------------------------

        best_face_id = None
        best_asd = None

        if face_candidates:

            identity_label, asd_score = max(
                face_candidates,
                key=lambda c: c[1]
            )

            best_face_id = self.resolve(identity_label)
            best_asd = float(asd_score)

            if best_face_id not in self.people:
                self.ensure_face_identity(best_face_id)

        if voice_speaker_label:
            self._turn_results_by_speaker.setdefault(voice_speaker_label, []).append({
            "top_face_id": best_face_id,
            "asd_score": best_asd,
        })

        best_face_sim = 1.0 if best_face_id else None

        visible_speaker = (
            best_asd is not None
            and best_asd >= ASD_LINK_THRESHOLD
        )

        # ------------------------------------------------------
        # Voice + confidently active face
        # ------------------------------------------------------

        if voice_id and visible_speaker:

            final_id = self.merge(
                voice_id,
                best_face_id
            )

            rec = self.people[final_id]

            rec.speaking_turns += 1

            return {
                "person_id": final_id,
                "label": rec.display_label(),

                "confirmation":
                    "DUAL_CONFIRMED"
                    if rec.is_confirmed(self.min_speaking_turns_for_confirmed)
                    else "LINKED_UNCONFIRMED",

                "confidence": round(
                    (
                        (voice_sim or 0.0)
                        + (best_face_sim or 0.0)
                        + best_asd
                    ) / 3,
                    4
                ),

                "voice_id": voice_id,
                "face_id": best_face_id,

                "asd_confidence": round(
                    best_asd,
                    4
                ),

                # NEW: expose every face TalkNet evaluated
                "face_candidates": [
                    {
                        "face_id": self.resolve(face_id),
                        "asd_confidence": round(
                            float(score),
                            4
                        )
                    }
                    for face_id, score in face_candidates
                ],

                "speaker_face_association":
                    self.speaker_face_associations().get(
                        voice_speaker_label
                    ),
            }

        # ------------------------------------------------------
        # Voice detected but no confidently active face
        # ------------------------------------------------------

        if voice_id and not visible_speaker:

            rec = self.people[voice_id]

            rec.voice_only_sightings += 1

            if best_face_id:
                self.people[
                    best_face_id
                ].face_only_sightings += 1

            return {
                "person_id": voice_id,
                "label": rec.display_label(),
                "confirmation": "VOICE_ONLY_OFFSCREEN",

                "confidence": round(
                    (voice_sim or 0.0) * 0.7,
                    4
                ),

                "voice_id": voice_id,
                "face_id": best_face_id,

                "asd_confidence": round(
                    best_asd or 0.0,
                    4
                ),

                "face_candidates": [
                    {
                        "face_id": self.resolve(face_id),
                        "asd_confidence": round(
                            float(score),
                            4
                        )
                    }
                    for face_id, score in face_candidates
                ],

                "speaker_face_association":
                    self.speaker_face_associations().get(
                        voice_speaker_label
                    ),
            }

        # ------------------------------------------------------
        # Face only
        # ------------------------------------------------------

        if best_face_id and not voice_id:

            rec = self.people[best_face_id]

            rec.face_only_sightings += 1

            return {
                "person_id": best_face_id,
                "label": rec.display_label(),
                "confirmation": "FACE_ONLY",

                "confidence": round(
                    (best_face_sim or 0.0) * 0.7,
                    4
                ),

                "voice_id": None,
                "face_id": best_face_id,

                "asd_confidence": round(
                    best_asd or 0.0,
                    4
                ),

                "face_candidates": [
                    {
                        "face_id": self.resolve(face_id),
                        "asd_confidence": round(
                            float(score),
                            4
                        )
                    }
                    for face_id, score in face_candidates
                ],

                "speaker_face_association":
                    self.speaker_face_associations().get(
                        voice_speaker_label
                    ),
            }

        # ------------------------------------------------------
        # Nothing usable
        # ------------------------------------------------------

        return {
            "person_id": None,
            "label": "Unknown",
            "confirmation": "UNCONFIRMED",
            "confidence": 0.0,

            "voice_id": None,
            "face_id": None,
            "asd_confidence": 0.0,

            "face_candidates": [],

            "speaker_face_association":
                self.speaker_face_associations().get(
                    voice_speaker_label
                ),
        }
    # ── explicit voice enrollment (the face reference belongs to Who Is This) ──
    def enroll_voice(self, name: str, embedding: torch.Tensor):
        rec = self._get_or_create_by_name(name)
        rec.voice_embedding, rec.voice_samples = embedding, 1
        rec.is_enrolled_voice = True

    # ── called later, by the caregiver/family naming flow ──
    def assign_name(self, person_id: str, name: str):
        pid = self.resolve(person_id)
        if pid in self.people:
            self.people[pid].name = name

    def confirmed_people(self) -> list:
        """People worth showing as real recurring participants (enrolled,
        or seen speaking >= MIN_SPEAKING_TURNS_FOR_CONFIRMED times) —
        useful for a 'who's in this video' summary view."""
        return [rec for rec in self.people.values() if rec.is_confirmed(self.min_speaking_turns_for_confirmed)]