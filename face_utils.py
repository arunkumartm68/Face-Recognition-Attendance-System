"""OpenCV / dlib helpers shared by the camera, feature-extraction and evaluation scripts."""
import csv
import hashlib
import json
import os
import time

import cv2
import numpy as np

import common

try:
    import dlib  # type: ignore
except ImportError as e:
    raise RuntimeError(
        "Missing dependency: dlib. Install the requirements first:  pip install -r requirements.txt"
    ) from e


def _dlib_path(filename):
    path = os.path.join(common.MODEL_DIR, filename)
    if path.isascii():
        return path
    # dlib opens files through narrow (ANSI) paths on Windows, which fails when
    # a folder name contains non-ASCII characters. A path relative to the
    # project folder avoids that.
    os.chdir(common.BASE_DIR)
    return os.path.relpath(path, common.BASE_DIR)


_models = None


def load_models():
    """Return (face detector, landmark predictor, face recognition model), loaded once."""
    global _models
    if _models is None:
        missing = common.missing_model_files()
        if missing:
            raise SystemExit("Missing dlib model file(s) in data/data_dlib/: " + ", ".join(missing)
                             + "\nDownload them with:  python download_models.py")
        _models = (dlib.get_frontal_face_detector(),
                   dlib.shape_predictor(_dlib_path(common.SHAPE_PREDICTOR_FILE)),
                   dlib.face_recognition_model_v1(_dlib_path(common.FACE_RECOGNITION_MODEL_FILE)))
    return _models


def detect_faces(detector, rgb):
    # Upsample once so smaller faces further from the camera are still found.
    # (A second pass without upsampling finds nothing extra: the upsampled
    # image pyramid already covers those scales.)
    return detector(rgb, 1)


def largest_face(faces):
    return max(faces, key=lambda rect: rect.width() * rect.height())


def face_descriptor(predictor, model, rgb, rect):
    """128D descriptor of the face in `rect`. `rgb` must be RGB - dlib's model was trained on RGB."""
    shape = predictor(rgb, rect)
    return np.array(model.compute_face_descriptor(rgb, shape), dtype=float)


def nearest(descriptor, known):
    """(index, distance) of the closest row of `known` (N x 128), or (None, None) if it is empty."""
    if len(known) == 0:
        return None, None
    distances = np.linalg.norm(known - descriptor, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def imread_rgb(path):
    """Load an image as RGB. Unlike cv2.imread this also works with non-ASCII paths on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, rgb):
    """Save an RGB image (format taken from the extension). Returns False on failure."""
    ok, buffer = cv2.imencode(os.path.splitext(path)[1] or '.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return False
    try:
        buffer.tofile(path)
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------
# features_all.csv - one averaged descriptor per registered student
# --------------------------------------------------------------------------

def load_features(csv_path=None):
    """Return (labels, N x 128 array) from features_all.csv."""
    labels, vectors = [], []
    try:
        with open(csv_path or common.FEATURES_CSV, newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) < 129:
                    continue
                try:
                    vector = [float(v) for v in row[1:129]]
                except ValueError:
                    continue
                labels.append(row[0].strip())
                vectors.append(vector)
    except FileNotFoundError:
        pass
    return labels, np.array(vectors, dtype=float).reshape(-1, 128)


def _replace_atomically(path, write):
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def photo_fingerprint():
    """Hash of every face photo's path, size and modification time.

    It changes whenever a photo or student folder is added, removed, renamed
    or replaced, without comparing clocks.
    """
    digest = hashlib.sha256()
    if os.path.isdir(common.FACES_DIR):
        for current, dirs, files in os.walk(common.FACES_DIR):
            dirs.sort()
            for name in sorted(files):
                if name.lower().endswith(common.IMAGE_EXTENSIONS):
                    path = os.path.join(current, name)
                    stat = os.stat(path)
                    entry = f'{os.path.relpath(path, common.FACES_DIR)}|{stat.st_size}|{stat.st_mtime_ns}\n'
                    digest.update(entry.encode('utf-8', 'replace'))
    return digest.hexdigest()


def write_features(rows, fingerprint):
    """Replace features_all.csv (and its metadata file) atomically.

    Extraction can be started by the register window, the dashboard and the
    scanner at the same time; writing a per-process temp file and swapping it
    in means concurrent runs never interleave rows and readers never see half
    a file. `fingerprint` describes the photos as they were when extraction
    started, so photos saved while it was running trigger another rebuild.
    The CSV is replaced before its metadata, so a crash in between only
    causes an extra rebuild.
    """
    os.makedirs(common.DATA_DIR, exist_ok=True)

    def write_csv(tmp):
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

    def write_meta(tmp):
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'version': common.FEATURES_VERSION, 'color': 'RGB', 'profiles': len(rows),
                       'photos_fingerprint': fingerprint,
                       'built_at': time.strftime(common.TIMESTAMP_FORMAT)}, f, indent=2)

    _replace_atomically(common.FEATURES_CSV, write_csv)
    _replace_atomically(common.FEATURES_META, write_meta)


def features_need_rebuild():
    """True if features_all.csv is missing, was built by an older version, or the photos changed since."""
    if not os.path.isfile(common.FEATURES_CSV):
        return True
    try:
        with open(common.FEATURES_META, encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return True     # written by an older version (descriptors computed on BGR images)
    if not isinstance(meta, dict) or meta.get('version') != common.FEATURES_VERSION:
        return True
    return meta.get('photos_fingerprint') != photo_fingerprint()
