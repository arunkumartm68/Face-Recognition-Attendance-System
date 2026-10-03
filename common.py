"""Shared configuration, database schema and helpers.

Used by the web app and by the camera / feature-extraction scripts, so it must
stay light: OpenCV and dlib helpers live in face_utils.py instead.
"""
import csv
import io
import os
import re
import secrets
import shutil
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Both locations can be overridden through the environment (the tests use a
# temporary folder so they never touch real data).
DATA_DIR = os.environ.get('ATTENDANCE_DATA_DIR') or os.path.join(BASE_DIR, 'data')
DB_PATH = os.environ.get('ATTENDANCE_DB') or os.path.join(BASE_DIR, 'attendance.db')

FACES_DIR = os.path.join(DATA_DIR, 'data_faces_from_camera')
FEATURES_CSV = os.path.join(DATA_DIR, 'features_all.csv')
FEATURES_META = os.path.join(DATA_DIR, 'features_meta.json')
LATENCY_JSON = os.path.join(DATA_DIR, 'registration_latency.json')
EVALUATION_JSON = os.path.join(DATA_DIR, 'evaluation_results.json')
SECRET_KEY_FILE = os.path.join(DATA_DIR, '.secret_key')

MODEL_DIR = os.path.join(BASE_DIR, 'data', 'data_dlib')
SHAPE_PREDICTOR_FILE = 'shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_FILE = 'dlib_face_recognition_resnet_model_v1.dat'
MODEL_FILES = (SHAPE_PREDICTOR_FILE, FACE_RECOGNITION_MODEL_FILE)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Largest Euclidean distance between two 128D face descriptors that still
# counts as the same person. Check it on your own data with evaluate_accuracy.py.
MATCH_THRESHOLD = float(os.environ.get('FACE_MATCH_THRESHOLD', '0.45'))
# Photos per student before the feature database is rebuilt automatically.
PHOTOS_PER_STUDENT = 10
LOW_ATTENDANCE_PERCENT = 75
# Bump whenever descriptors change, so old features_all.csv files get rebuilt.
FEATURES_VERSION = 2  # 2 = descriptors computed on RGB images

GENERAL_SUBJECT = 'General'
STATUS_PRESENT, STATUS_LATE, STATUS_ABSENT = 'Present', 'Late', 'Absent'

DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%I:%M:%S %p'            # attendance.time, e.g. 02:30:15 PM
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

DEFAULT_ADMIN_USERNAME = 'admin'
DEFAULT_ADMIN_PASSWORD = 'admin'
MIN_PASSWORD_LENGTH = 6

ATTENDANCE_DDL = '''
    CREATE TABLE attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        register_number TEXT NOT NULL,
        name TEXT,
        time TEXT,
        date TEXT,
        subject TEXT DEFAULT 'General',
        status TEXT NOT NULL DEFAULT 'Present',
        UNIQUE(register_number, date, subject)
    )
'''


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def connect(db_path=None):
    """SQLite connection with dict-style rows. Waits up to 10 s while another process writes."""
    conn = sqlite3.connect(db_path or DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn, table):
    return [row[1] for row in conn.execute(f'PRAGMA table_info({table})')]


def init_db(db_path=None):
    """Create missing tables and migrate databases made by older versions."""
    conn = connect(db_path)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS students (
                register_number TEXT PRIMARY KEY,
                name TEXT,
                password TEXT
            )
        ''')

        staff_cols = _columns(conn, 'staff')
        if not staff_cols:
            conn.execute('''
                CREATE TABLE staff (
                    subject_name TEXT PRIMARY KEY,
                    professor_name TEXT DEFAULT '',
                    password TEXT
                )
            ''')
        elif 'professor_name' not in staff_cols:
            conn.execute("ALTER TABLE staff ADD COLUMN professor_name TEXT DEFAULT ''")

        conn.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')

        # Attendance is keyed on register_number (stable) rather than the old
        # display string "<reg>_<name>", so renaming a student keeps history.
        columns = _columns(conn, 'attendance')
        if not columns:
            conn.execute(ATTENDANCE_DDL)
        elif 'register_number' not in columns:
            conn.execute('ALTER TABLE attendance RENAME TO attendance_old')
            conn.execute(ATTENDANCE_DDL)
            subject_expr = 'subject' if 'subject' in columns else "'General'"
            conn.execute('''
                INSERT OR IGNORE INTO attendance (register_number, name, time, date, subject)
                SELECT
                    CASE WHEN instr(name, '_') > 0
                         THEN substr(name, 1, instr(name, '_') - 1)
                         ELSE name END,
                    CASE WHEN instr(name, '_') > 0
                         THEN substr(name, instr(name, '_') + 1)
                         ELSE name END,
                    time, date, ''' + subject_expr + '''
                FROM attendance_old
            ''')
            conn.execute('DROP TABLE attendance_old')
        elif 'status' not in columns:
            conn.execute("ALTER TABLE attendance ADD COLUMN status TEXT NOT NULL DEFAULT 'Present'")

        conn.execute('''
            CREATE TABLE IF NOT EXISTS enrollments (
                register_number TEXT NOT NULL,
                subject_name TEXT NOT NULL,
                PRIMARY KEY (register_number, subject_name)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                date TEXT NOT NULL,
                started_at TEXT NOT NULL,
                planned_end TEXT,
                ended_at TEXT,
                duration_min REAL DEFAULT 0,
                late_after_min REAL DEFAULT 0,
                started_by TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                actor_role TEXT,
                actor TEXT,
                action TEXT NOT NULL,
                register_number TEXT,
                student_name TEXT,
                subject TEXT,
                date TEXT,
                details TEXT
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_attendance_subject_date ON attendance(subject, date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_enrollments_subject ON enrollments(subject_name)')

        # Older versions stored passwords in plain text.
        for table, key in (('students', 'register_number'), ('staff', 'subject_name'), ('admins', 'username')):
            for row in conn.execute(f'SELECT {key}, password FROM {table}').fetchall():
                if row['password'] and not is_password_hash(row['password']):
                    conn.execute(f'UPDATE {table} SET password = ? WHERE {key} = ?',
                                 (hash_password(row['password']), row[key]))

        if conn.execute('SELECT COUNT(*) FROM admins').fetchone()[0] == 0:
            conn.execute('INSERT INTO admins (username, password) VALUES (?, ?)',
                         (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD)))
        conn.commit()
    finally:
        conn.close()


def missing_model_files():
    """Names of the dlib model files that have not been downloaded yet."""
    return [name for name in MODEL_FILES if not os.path.isfile(os.path.join(MODEL_DIR, name))]


# --------------------------------------------------------------------------
# Passwords and validation
# --------------------------------------------------------------------------

def hash_password(password):
    return generate_password_hash(password)


def is_password_hash(value):
    return (isinstance(value, str) and value.startswith(('scrypt:', 'pbkdf2:'))
            and value.count('$') == 2)


def verify_password(stored, password):
    """Check a password against its stored hash.

    A plain-text value (left by a very old database) is still accepted so the
    caller can replace it with a hash.
    """
    if not stored or not password:
        return False
    if is_password_hash(stored):
        return check_password_hash(stored, password)
    return secrets.compare_digest(stored.encode(), password.encode())


def validate_password(password):
    if len(password or '') < MIN_PASSWORD_LENGTH:
        return f'Password must be at least {MIN_PASSWORD_LENGTH} characters.'
    return None


# Characters Windows does not allow in folder names. The register number and
# name together become the face-photo folder name "person_<n>_<reg>_<name>".
_FORBIDDEN_CHARS = set('\\/:*?"<>|')


def _has_bad_chars(value):
    return bool(_FORBIDDEN_CHARS & set(value)) or any(ord(c) < 32 for c in value)


def validate_register_number(reg):
    if not reg:
        return 'Register number is required.'
    if len(reg) > 50:
        return 'Register number is too long (max 50 characters).'
    # The profile label "<reg>_<name>" is split on its first '_' when a face is
    # recognised, so the register number itself must not contain one.
    if '_' in reg:
        return "Register number cannot contain '_'."
    if _has_bad_chars(reg):
        return 'Register number cannot contain \\ / : * ? " < > |'
    return None


def validate_name(name):
    if not name:
        return 'Name is required.'
    if len(name) > 100:
        return 'Name is too long (max 100 characters).'
    if _has_bad_chars(name):
        return 'Name cannot contain \\ / : * ? " < > |'
    return None


def validate_subject_name(subject):
    if not subject:
        return 'Subject name is required.'
    if len(subject) > 60:
        return 'Subject name is too long (max 60 characters).'
    if subject.lower() == GENERAL_SUBJECT.lower():
        return f'"{GENERAL_SUBJECT}" is reserved for attendance taken without a subject.'
    return None


# --------------------------------------------------------------------------
# Face-photo folders and profile labels
# --------------------------------------------------------------------------

_FOLDER_RE = re.compile(r'^person_\d+_(.+)$')


def folder_label(folder_name):
    """Profile label for a photo folder: 'person_3_21CS001_Arun' -> '21CS001_Arun'."""
    match = _FOLDER_RE.match(folder_name)
    return match.group(1) if match else folder_name


def split_label(label):
    """Split a profile label "<reg>_<name>" into (register_number, name)."""
    reg, sep, name = label.partition('_')
    return (reg, name) if sep else (label, label)


def _face_folders():
    if not os.path.isdir(FACES_DIR):
        return []
    return [entry for entry in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, entry))]


def face_photo_counts():
    """{register_number: number of saved face photos}."""
    counts = {}
    for entry in _face_folders():
        reg = split_label(folder_label(entry))[0]
        photos = [f for f in os.listdir(os.path.join(FACES_DIR, entry))
                  if f.lower().endswith(IMAGE_EXTENSIONS)]
        counts[reg] = counts.get(reg, 0) + len(photos)
    return counts


def delete_face_folders(register_number):
    """Delete every face-photo folder of a student. Returns how many were removed."""
    removed = 0
    for entry in _face_folders():
        if split_label(folder_label(entry))[0] == register_number:
            shutil.rmtree(os.path.join(FACES_DIR, entry), ignore_errors=True)
            removed += 1
    return removed


# --------------------------------------------------------------------------
# Subjects, rosters and attendance changes
# --------------------------------------------------------------------------

def subject_names(conn):
    return [row[0] for row in conn.execute(
        'SELECT subject_name FROM staff ORDER BY subject_name COLLATE NOCASE')]


def is_known_subject(conn, subject):
    return subject == GENERAL_SUBJECT or conn.execute(
        'SELECT 1 FROM staff WHERE subject_name = ?', (subject,)).fetchone() is not None


def all_subjects(conn):
    """Subjects that have reports: every staff subject, plus General once it has been used."""
    names = subject_names(conn)
    if conn.execute('SELECT 1 FROM attendance WHERE subject = ? LIMIT 1', (GENERAL_SUBJECT,)).fetchone():
        names.insert(0, GENERAL_SUBJECT)
    return names


def has_enrollments(conn, subject):
    return conn.execute('SELECT 1 FROM enrollments WHERE subject_name = ? LIMIT 1',
                        (subject,)).fetchone() is not None


def roster(conn, subject):
    """Students expected in a subject.

    General, and any subject nobody has been assigned to yet, expect every
    student; otherwise only the assigned students.
    """
    if subject != GENERAL_SUBJECT and has_enrollments(conn, subject):
        return conn.execute('''
            SELECT s.register_number, s.name FROM students s
            JOIN enrollments e ON e.register_number = s.register_number
            WHERE e.subject_name = ?
            ORDER BY s.register_number
        ''', (subject,)).fetchall()
    return conn.execute('SELECT register_number, name FROM students ORDER BY register_number').fetchall()


def on_roster(conn, register_number, subject):
    if subject == GENERAL_SUBJECT or not has_enrollments(conn, subject):
        return True
    return conn.execute('SELECT 1 FROM enrollments WHERE subject_name = ? AND register_number = ?',
                        (subject, register_number)).fetchone() is not None


def subjects_for_student(conn, register_number):
    """Subjects shown to a student: the ones they are expected in or have records for."""
    with_records = {row[0] for row in conn.execute(
        'SELECT DISTINCT subject FROM attendance WHERE register_number = ?', (register_number,))}
    return [s for s in all_subjects(conn)
            if s in with_records or on_roster(conn, register_number, s)]


def log_action(conn, actor_role, actor, action, register_number=None, student_name=None,
               subject=None, date=None, details=''):
    conn.execute('''
        INSERT INTO audit_log (timestamp, actor_role, actor, action, register_number,
                               student_name, subject, date, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime(TIMESTAMP_FORMAT), actor_role, actor, action,
          register_number, student_name, subject, date, details))


def set_attendance_status(conn, register_number, subject, date, status, actor_role, actor):
    """Manually set one student's status for one class day and record it in the audit log.

    Returns the previous status, or None when the student does not exist.
    """
    student = conn.execute('SELECT name FROM students WHERE register_number = ?',
                           (register_number,)).fetchone()
    if student is None:
        return None
    key = (register_number, date, subject)
    row = conn.execute('SELECT status FROM attendance WHERE register_number = ? AND date = ? AND subject = ?',
                       key).fetchone()
    previous = row['status'] if row else STATUS_ABSENT
    if previous == status:
        return previous
    if status == STATUS_ABSENT:
        conn.execute('DELETE FROM attendance WHERE register_number = ? AND date = ? AND subject = ?', key)
    elif row:
        conn.execute('UPDATE attendance SET status = ?, name = ? '
                     'WHERE register_number = ? AND date = ? AND subject = ?',
                     (status, student['name']) + key)
    else:
        conn.execute('INSERT INTO attendance (register_number, name, time, date, subject, status) '
                     'VALUES (?, ?, ?, ?, ?, ?)',
                     (register_number, student['name'], datetime.now().strftime(TIME_FORMAT),
                      date, subject, status))
    log_action(conn, actor_role, actor, f'Marked {status}', register_number=register_number,
               student_name=student['name'], subject=subject, date=date, details=f'was {previous}')
    conn.commit()
    return previous


# --------------------------------------------------------------------------
# Bulk import
# --------------------------------------------------------------------------

_IMPORT_COLUMNS = {
    'register_number': ('register_number', 'register_no', 'reg_no', 'regno', 'reg_number',
                        'registration_number', 'roll_no', 'roll_number'),
    'name': ('name', 'student_name', 'full_name'),
    'password': ('password', 'pass', 'pwd'),
    'subjects': ('subjects', 'subject'),
}
MAX_IMPORT_ROWS = 5000


def import_students_csv(conn, text):
    """Add or update students from CSV text (first row = header).

    Returns {'added', 'updated', 'enrolled', 'errors': [message, ...]}. Bad rows
    are reported and skipped; the rest are imported.
    """
    result = {'added': 0, 'updated': 0, 'enrolled': 0, 'errors': []}
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        result['errors'].append('The file is empty.')
        return result
    normalized = [re.sub(r'[\s\-]+', '_', h.strip().lower()) for h in header]
    index = {}
    for field, aliases in _IMPORT_COLUMNS.items():
        for i, column in enumerate(normalized):
            if column in aliases:
                index[field] = i
                break
    if 'register_number' not in index or 'name' not in index:
        result['errors'].append('The first row must be a header with at least '
                                '"register_number" and "name" columns.')
        return result

    subjects = set(subject_names(conn))
    existing = {row[0] for row in conn.execute('SELECT register_number FROM students')}
    seen = set()
    rows_read = 0
    for line_no, row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in row):
            continue
        rows_read += 1
        if rows_read > MAX_IMPORT_ROWS:
            result['errors'].append(f'Stopped after {MAX_IMPORT_ROWS} rows.')
            break

        def cell(field):
            i = index.get(field)
            return row[i].strip() if i is not None and i < len(row) else ''

        reg, name, password = cell('register_number'), cell('name'), cell('password')
        error = validate_register_number(reg) or validate_name(name)
        if not error and reg in seen:
            error = 'register number appears more than once in this file'
        if not error:
            if password:
                error = validate_password(password)
            elif reg not in existing:
                error = 'a password is required for new students'
        if error:
            result['errors'].append(f'Line {line_no}: {error}')
            continue

        seen.add(reg)
        if reg in existing:
            if password:
                conn.execute('UPDATE students SET name = ?, password = ? WHERE register_number = ?',
                             (name, hash_password(password), reg))
            else:
                conn.execute('UPDATE students SET name = ? WHERE register_number = ?', (name, reg))
            result['updated'] += 1
        else:
            conn.execute('INSERT INTO students (register_number, name, password) VALUES (?, ?, ?)',
                         (reg, name, hash_password(password)))
            existing.add(reg)
            result['added'] += 1

        for subject in (s.strip() for s in re.split(r'[;|]', cell('subjects'))):
            if not subject:
                continue
            if subject not in subjects:
                result['errors'].append(f'Line {line_no}: unknown subject "{subject}" '
                                        '(student imported, subject skipped)')
                continue
            cur = conn.execute('INSERT OR IGNORE INTO enrollments (register_number, subject_name) '
                               'VALUES (?, ?)', (reg, subject))
            result['enrolled'] += cur.rowcount
    conn.commit()
    return result
