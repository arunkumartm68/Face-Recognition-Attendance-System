"""Point the app at a throw-away data folder before anything imports common.py.

Every test module imports this first, so the tests never touch the real
attendance.db, face photos or secret key.
"""
import atexit
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
TMP = tempfile.mkdtemp(prefix='attendance-tests-')

os.environ['ATTENDANCE_DATA_DIR'] = os.path.join(TMP, 'data')
os.environ['ATTENDANCE_DB'] = os.path.join(TMP, 'attendance.db')
os.environ.pop('SECRET_KEY', None)
os.environ.pop('FACE_MATCH_THRESHOLD', None)
os.makedirs(os.environ['ATTENDANCE_DATA_DIR'], exist_ok=True)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
atexit.register(shutil.rmtree, TMP, ignore_errors=True)


def reset_state():
    """Fresh database and empty data folder for each test."""
    import common
    if os.path.exists(common.DB_PATH):
        os.remove(common.DB_PATH)
    shutil.rmtree(common.FACES_DIR, ignore_errors=True)
    for path in (common.FEATURES_CSV, common.FEATURES_META, common.LATENCY_JSON, common.EVALUATION_JSON):
        if os.path.exists(path):
            os.remove(path)
    os.makedirs(common.FACES_DIR, exist_ok=True)
    common.init_db()
