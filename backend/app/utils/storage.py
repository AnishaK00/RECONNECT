from pathlib import Path
from datetime import datetime
from threading import Lock

BASE_DIR = Path(__file__).resolve().parents[2] #back to backend directory

UPLOAD_ROOT = BASE_DIR / "uploads"
# ORIGINAL_FOLDER = UPLOAD_ROOT / "original"
# CLEANED_FOLDER = UPLOAD_ROOT / "cleaned"
STREAMS_FOLDER = UPLOAD_ROOT / "streams"
# SEGMENTS_FOLDER = UPLOAD_ROOT / "segments"

# ORIGINAL_FOLDER.mkdir(parents=True, exist_ok=True)
# CLEANED_FOLDER.mkdir(parents=True, exist_ok=True)
STREAMS_FOLDER.mkdir(parents=True, exist_ok=True)
_SESSION_LOCK = Lock()
_NEXT_SESSION_NUMBER = 1
# SEGMENTS_FOLDER.mkdir(parents=True, exist_ok=True)


# def get_original_folder() -> Path:
#     return ORIGINAL_FOLDER


# def get_cleaned_folder() -> Path:
#     return CLEANED_FOLDER


def get_stream_session_folder(session_id: str) -> Path:
    """Return the private storage root for one live audio stream."""

    return STREAMS_FOLDER / session_id


def allocate_stream_session_id() -> str:
    """Allocate the next compact stream-session directory name."""

    global _NEXT_SESSION_NUMBER
    with _SESSION_LOCK:
        while True:
            session_id = f"session_{_NEXT_SESSION_NUMBER:03d}"
            _NEXT_SESSION_NUMBER += 1
            if not (STREAMS_FOLDER / session_id).exists():
                return session_id


# def get_segments_root() -> Path:
#     return SEGMENTS_FOLDER


# def get_fixtures_folder() -> Path:
#     return FIXTURES_FOLDER


def generate_filename(original_filename: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = Path(original_filename).suffix.lower()
    return f"{timestamp}{extension}"


# def get_session_id(stored_filename: str) -> str:
#     return Path(stored_filename).stem


# def get_original_audio_path(filename: str) -> Path:
#     return ORIGINAL_FOLDER / filename


# def get_cleaned_audio_path(stored_filename: str) -> Path:
#     filename_without_ext = Path(stored_filename).stem
#     return CLEANED_FOLDER / f"{filename_without_ext}.wav"


# def get_segments_folder(session_id: str) -> Path:
#     return SEGMENTS_FOLDER / session_id
