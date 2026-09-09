import sqlite3
import threading
import hashlib
import hmac
import secrets
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_DB_LOCK = threading.Lock()
_DB_INITIALISED = False

DATABASE_PATH = Path(__file__).resolve().parent / "reconnect.db"
FACE_MATCH_THRESHOLD = 0.50


def _enable_wal_once():
    """Enable WAL journal mode exactly once at module load.

    WAL mode requires an *exclusive* lock on the database file, so it must
    be set before any other connections are opened.  Calling it from inside
    _connect() (which can be called concurrently) causes 'database is locked'.
    Errors are silently ignored – the database still works in the default
    DELETE journal mode if WAL cannot be activated.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
    except sqlite3.OperationalError:
        pass  # Already in WAL or some other process holds the DB; non-fatal


_enable_wal_once()


def embedding_to_blob(embedding):
    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_to_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32).copy()


def voice_blob_to_embeddings(blob, dimension=192):
    values = blob_to_embedding(blob)
    if values.size % dimension != 0:
        return []
    return [
        values[index:index + dimension].copy()
        for index in range(0, values.size, dimension)
    ]


def image_to_blob(image_input):
    """Converts an OpenCV image (numpy array), a file path, or raw bytes into a JPEG binary blob."""
    if image_input is None:
        return None
    if isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input)
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.exists() and p.is_file():
            with open(p, "rb") as f:
                return f.read()
        return None
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            return None
        import cv2
        success, encoded = cv2.imencode(".jpg", image_input)
        if success:
            return encoded.tobytes()
    return None


def blob_to_image(blob_data):
    """Decodes JPEG/PNG bytes from SQLite into an OpenCV BGR image (numpy array)."""
    if not blob_data:
        return None
    import cv2
    buf = np.frombuffer(blob_data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@contextmanager
def _connect():
    """Open a thread-safe SQLite connection with automatic closing and busy timeout."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_database():
    global _DB_INITIALISED
    if _DB_INITIALISED:
        return
    with _DB_LOCK, _connect() as connection:
        existing = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'enrolled_identities'"
        ).fetchone()

        legacy = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'identities'"
        ).fetchone()

        if not existing and legacy:
            connection.execute(
                "ALTER TABLE identities RENAME TO enrolled_identities"
            )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS enrolled_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                name TEXT NOT NULL,
                relation TEXT NOT NULL,
                face_embedding BLOB NOT NULL,
                voice_embedding BLOB,
                face_image BLOB
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS unenrolled_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                face_embedding BLOB,
                voice_embedding BLOB,
                face_image BLOB
            )
            """
        )

        # Migration: ensure face_image column exists on existing tables
        enrolled_cols = [r[1] for r in connection.execute("PRAGMA table_info(enrolled_identities)").fetchall()]
        if "face_image" not in enrolled_cols:
            connection.execute("ALTER TABLE enrolled_identities ADD COLUMN face_image BLOB")
        if "patient_id" not in enrolled_cols:
            connection.execute("ALTER TABLE enrolled_identities ADD COLUMN patient_id INTEGER")

        connection.execute("""
            CREATE TABLE IF NOT EXISTS caregivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE, phone TEXT NOT NULL UNIQUE,
                password_salt BLOB NOT NULL, password_hash BLOB NOT NULL,
                caregiver_type TEXT NOT NULL, family_relation TEXT, profession TEXT,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT, caregiver_id INTEGER NOT NULL,
                full_name TEXT NOT NULL, patient_code TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE, password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL, dob TEXT NOT NULL, phone TEXT NOT NULL,
                email TEXT NOT NULL, address TEXT NOT NULL, medical_information TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (caregiver_id) REFERENCES caregivers(id) ON DELETE CASCADE
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, actor_type TEXT NOT NULL,
                actor_id INTEGER NOT NULL, token_hash BLOB NOT NULL UNIQUE,
                expires_at TEXT NOT NULL, created_at TEXT NOT NULL
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_identities_patient ON enrolled_identities(patient_id)")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS pending_unknowns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                face_image BLOB NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                enrolled_identity_id INTEGER,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                FOREIGN KEY (enrolled_identity_id) REFERENCES enrolled_identities(id)
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_pending_unknowns_patient ON pending_unknowns(patient_id, resolved_at)")

        unenrolled_cols = [r[1] for r in connection.execute("PRAGMA table_info(unenrolled_identities)").fetchall()]
        if "face_image" not in unenrolled_cols:
            connection.execute("ALTER TABLE unenrolled_identities ADD COLUMN face_image BLOB")

    _DB_INITIALISED = True


def get_all_identities(patient_id=None):
    init_database()
    with _DB_LOCK, _connect() as connection:
        if patient_id is None:
            return connection.execute("SELECT * FROM enrolled_identities ORDER BY id").fetchall()
        return connection.execute("SELECT * FROM enrolled_identities WHERE patient_id = ? ORDER BY id", (patient_id,)).fetchall()


def get_identity_by_name(name):
    init_database()
    with _DB_LOCK, _connect() as connection:
        return connection.execute(
            "SELECT * FROM enrolled_identities WHERE name = ? "
            "ORDER BY id LIMIT 1",
            (name,),
        ).fetchone()


def find_matching_face(embedding, threshold=FACE_MATCH_THRESHOLD, patient_id=None):
    candidate = np.asarray(embedding, dtype=np.float32)
    best_row = None
    best_similarity = -1.0

    for row in get_all_identities(patient_id=patient_id):
        stored = blob_to_embedding(row["face_embedding"])
        if stored.shape != candidate.shape:
            continue
        similarity = float(np.dot(candidate, stored))
        if similarity > best_similarity:
            best_similarity = similarity
            best_row = row

    if best_row is not None and best_similarity >= threshold:
        return best_row, best_similarity
    return None, best_similarity


def create_identity(name, relation, face_embedding, face_image=None, patient_id=None):
    init_database()
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO enrolled_identities
                (timestamp, name, relation, face_embedding, voice_embedding, face_image, patient_id)
            VALUES (?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                name,
                relation,
                embedding_to_blob(face_embedding),
                image_to_blob(face_image),
                patient_id,
            ),
        )
        return cursor.lastrowid


_PASSWORD_ITERATIONS = 310_000


def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PASSWORD_ITERATIONS)


def _new_password_fields(password):
    salt = secrets.token_bytes(16)
    return salt, _hash_password(password, salt)


def create_caregiver(*, name, email, phone, password, caregiver_type, family_relation=None, profession=None):
    init_database()
    salt, password_hash = _new_password_fields(password)
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute("""INSERT INTO caregivers
            (name, email, phone, password_salt, password_hash, caregiver_type, family_relation, profession, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, email.lower(), phone, salt, password_hash, caregiver_type, family_relation, profession, datetime.now(timezone.utc).isoformat()))
        return cursor.lastrowid


def create_patient(*, caregiver_phone, full_name, username, password, dob, phone, email, address, medical_information):
    init_database()
    salt, password_hash = _new_password_fields(password)
    with _DB_LOCK, _connect() as connection:
        caregiver = connection.execute("SELECT id FROM caregivers WHERE phone = ?", (caregiver_phone,)).fetchone()
        if not caregiver:
            return None
        cursor = connection.execute("""INSERT INTO patients
            (caregiver_id, full_name, patient_code, username, password_salt, password_hash, dob, phone, email, address, medical_information, created_at)
            VALUES (?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (caregiver["id"], full_name, username, salt, password_hash, dob, phone, email.lower(), address, medical_information, datetime.now(timezone.utc).isoformat()))
        patient_id = cursor.lastrowid
        patient_code = f"PT{patient_id:04d}"
        connection.execute("UPDATE patients SET patient_code = ? WHERE id = ?", (patient_code, patient_id))
        return patient_id


def authenticate_actor(actor_type, identifier, password):
    init_database()
    table, column = ("caregivers", "email") if actor_type == "caregiver" else ("patients", "username")
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(f"SELECT * FROM {table} WHERE {column} = ?", (identifier.lower() if actor_type == "caregiver" else identifier,)).fetchone()
        if not row or not hmac.compare_digest(_hash_password(password, row["password_salt"]), row["password_hash"]):
            return None
        return dict(row)


def create_auth_session(actor_type, actor_id, lifetime_days=7):
    init_database()
    token = secrets.token_urlsafe(48)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    now = datetime.now(timezone.utc)
    with _DB_LOCK, _connect() as connection:
        connection.execute("INSERT INTO auth_sessions (actor_type, actor_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?)",
            (actor_type, actor_id, digest, (now + timedelta(days=lifetime_days)).isoformat(), now.isoformat()))
    return token


def get_session_actor(token):
    init_database()
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute("SELECT actor_type, actor_id, expires_at FROM auth_sessions WHERE token_hash = ?", (digest,)).fetchone()
        if not row or datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            return None
        return dict(row)


def get_caregiver_patients(caregiver_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        return [dict(row) for row in connection.execute("SELECT id, full_name, patient_code FROM patients WHERE caregiver_id = ? ORDER BY id", (caregiver_id,)).fetchall()]


def caregiver_owns_patient(caregiver_id, patient_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        return connection.execute("SELECT 1 FROM patients WHERE id = ? AND caregiver_id = ?", (patient_id, caregiver_id)).fetchone() is not None


def get_patient_enrolled_people(patient_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        return [dict(row) for row in connection.execute("SELECT id, patient_id, name, relation FROM enrolled_identities WHERE patient_id = ? ORDER BY id", (patient_id,)).fetchall()]


def get_patient_identity(identity_id, patient_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute("SELECT * FROM enrolled_identities WHERE id = ? AND patient_id = ?", (identity_id, patient_id)).fetchone()
        return dict(row) if row else None


def get_patient_identity_by_name(name, patient_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute("SELECT * FROM enrolled_identities WHERE name = ? AND patient_id = ? ORDER BY id LIMIT 1", (name, patient_id)).fetchone()
        return dict(row) if row else None


def get_patient_identity_image(identity_id, patient_id):
    row = get_patient_identity(identity_id, patient_id)
    return row["face_image"] if row else None


def create_pending_unknown(patient_id, face_image):
    image_blob = image_to_blob(face_image)
    if not image_blob:
        return None
    init_database()
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute("INSERT INTO pending_unknowns (patient_id, face_image, created_at) VALUES (?, ?, ?)",
            (patient_id, image_blob, datetime.now(timezone.utc).isoformat()))
        return cursor.lastrowid


def get_pending_unknowns(patient_id, include_resolved=False):
    init_database()
    where = "patient_id = ?" if include_resolved else "patient_id = ? AND resolved_at IS NULL"
    with _DB_LOCK, _connect() as connection:
        return [dict(row) for row in connection.execute(
            f"SELECT id, patient_id, created_at, resolved_at, enrolled_identity_id FROM pending_unknowns WHERE {where} ORDER BY id DESC", (patient_id,)).fetchall()]


def get_pending_unknown_image(pending_id, patient_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute("SELECT face_image FROM pending_unknowns WHERE id = ? AND patient_id = ?", (pending_id, patient_id)).fetchone()
        return row["face_image"] if row else None


def resolve_pending_unknown(pending_id, patient_id, enrolled_identity_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute("UPDATE pending_unknowns SET resolved_at = ?, enrolled_identity_id = ? WHERE id = ? AND patient_id = ? AND resolved_at IS NULL",
            (datetime.now(timezone.utc).isoformat(), enrolled_identity_id, pending_id, patient_id))


def delete_identity(identity_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute("DELETE FROM enrolled_identities WHERE id = ?", (identity_id,))


def update_voice_embedding(identity_id, voice_embedding):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE enrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(voice_embedding), identity_id),
        )


def append_voice_embedding(identity_id, voice_embedding):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT voice_embedding FROM enrolled_identities WHERE id = ?",
            (identity_id,),
        ).fetchone()
        existing = row["voice_embedding"] if row else None
        current = blob_to_embedding(existing) if existing else np.array([], dtype=np.float32)
        combined = np.concatenate((current, np.asarray(voice_embedding, dtype=np.float32)))
        connection.execute(
            "UPDATE enrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(combined), identity_id),
        )

# ---------- Unenrolled identities helpers ----------

def create_unenrolled_identity(face_embedding=None, voice_embedding=None, face_image=None):
    """Create a new unenrolled identity with optional face and voice embeddings and face image.
    Returns the generated row id.
    """
    init_database()
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO unenrolled_identities (timestamp, face_embedding, voice_embedding, face_image)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                embedding_to_blob(face_embedding) if face_embedding is not None else None,
                embedding_to_blob(voice_embedding) if voice_embedding is not None else None,
                image_to_blob(face_image),
            ),
        )
        return cursor.lastrowid

def get_unenrolled_identity(un_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT * FROM unenrolled_identities WHERE id = ?",
            (int(un_id),),
        ).fetchone()
        return dict(row) if row else None

def update_unenrolled_face(un_id, face_embedding, face_image=None):
    """Updates face embedding and optionally the best face image for an unenrolled identity."""
    init_database()
    img_blob = image_to_blob(face_image)
    with _DB_LOCK, _connect() as connection:
        if img_blob is not None:
            connection.execute(
                "UPDATE unenrolled_identities SET face_embedding = ?, face_image = ? WHERE id = ?",
                (embedding_to_blob(face_embedding), img_blob, int(un_id)),
            )
        else:
            connection.execute(
                "UPDATE unenrolled_identities SET face_embedding = ? WHERE id = ?",
                (embedding_to_blob(face_embedding), int(un_id)),
            )

def update_unenrolled_voice(un_id, voice_embedding):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE unenrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(voice_embedding), un_id),
        )

def get_identity_image(identity_id, as_cv2=False):
    """Fetch the face image for an enrolled identity.
    Returns raw bytes by default, or an OpenCV numpy image if as_cv2=True.
    """
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT face_image FROM enrolled_identities WHERE id = ?",
            (int(identity_id),),
        ).fetchone()
        if not row or not row["face_image"]:
            return None
        blob = row["face_image"]
        return blob_to_image(blob) if as_cv2 else blob

def get_unenrolled_identity_image(un_id, as_cv2=False):
    """Fetch the face image for an unenrolled identity.
    Returns raw bytes by default, or an OpenCV numpy image if as_cv2=True.
    """
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT face_image FROM unenrolled_identities WHERE id = ?",
            (int(un_id),),
        ).fetchone()
        if not row or not row["face_image"]:
            return None
        blob = row["face_image"]
        return blob_to_image(blob) if as_cv2 else blob

def find_matching_enrolled_voice(embedding, threshold=0.50):
    """Search enrolled identities for a voice embedding similarity above threshold.
    Returns (row_dict, similarity) or (None, -1.0).
    """
    candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
    cand_norm = np.linalg.norm(candidate)
    if cand_norm == 0:
        return None, -1.0
    best_row = None
    best_similarity = -1.0
    with _DB_LOCK, _connect() as connection:
        rows = connection.execute(
            "SELECT id, name, relation, voice_embedding FROM enrolled_identities WHERE voice_embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            voice_blob = row["voice_embedding"]
            if voice_blob is None:
                continue
            for stored_emb in voice_blob_to_embeddings(voice_blob):
                stored = np.asarray(stored_emb, dtype=np.float32).reshape(-1)
                stored_norm = np.linalg.norm(stored)
                if stored.shape != candidate.shape or stored_norm == 0:
                    continue
                sim = float(np.dot(candidate, stored) / (cand_norm * stored_norm))
                if sim > best_similarity:
                    best_similarity = sim
                    best_row = dict(row)
    if best_row is not None and best_similarity >= threshold:
        return best_row, best_similarity
    return None, best_similarity


def find_matching_unenrolled_voice(embedding, threshold=0.42):
    """Search unenrolled identities for a voice embedding similarity above threshold.
    Returns (row_dict, similarity) or (None, -1.0).
    """
    candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
    cand_norm = np.linalg.norm(candidate)
    if cand_norm == 0:
        return None, -1.0
    best_row = None
    best_similarity = -1.0
    with _DB_LOCK, _connect() as connection:
        rows = connection.execute("SELECT * FROM unenrolled_identities WHERE voice_embedding IS NOT NULL").fetchall()
        for row in rows:
            voice_blob = row["voice_embedding"]
            if voice_blob is None:
                continue
            for stored_emb in voice_blob_to_embeddings(voice_blob):
                stored = np.asarray(stored_emb, dtype=np.float32).reshape(-1)
                stored_norm = np.linalg.norm(stored)
                if stored.shape != candidate.shape or stored_norm == 0:
                    continue
                sim = float(np.dot(candidate, stored) / (cand_norm * stored_norm))
                if sim > best_similarity:
                    best_similarity = sim
                    best_row = dict(row)
    if best_row is not None and best_similarity >= threshold:
        return best_row, best_similarity
    return None, best_similarity
