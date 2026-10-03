"""Download dlib's two pre-trained models into data/data_dlib/.

    python download_models.py

About 85 MB is downloaded from dlib.net (about 122 MB once unpacked). The
server can be slow; dropped connections are resumed automatically and files
that are already present and intact are skipped.
"""
import bz2
import hashlib
import http.client
import os
import sys
import time
import urllib.error
import urllib.request

from common import FACE_RECOGNITION_MODEL_FILE, MODEL_DIR, SHAPE_PREDICTOR_FILE

MODELS = {
    SHAPE_PREDICTOR_FILE: (
        'https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f'),
    FACE_RECOGNITION_MODEL_FILE: (
        'https://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
        '55533b28a95800a551ba546ba62fe69625c7e95a7061c338adffead08719da30'),
}
CHUNK = 1 << 20
ATTEMPTS = 8


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path):
    """Download `url` to `path`, resuming after dropped connections. True when complete."""
    for attempt in range(1, ATTEMPTS + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        total = None
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:     # server ignored the range: start over
                    have = 0
                length = response.headers.get('Content-Length')
                total = have + int(length) if length else None
                with open(path, 'ab' if have else 'wb') as out:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        if total:
                            print(f"\r  {have / 1e6:6.1f} / {total / 1e6:.1f} MB", end='', flush=True)
            # A dropped connection just ends the stream early, so check the size.
            if total is None or have >= total:
                print()
                return True
            print(f"\n  connection dropped at {have / 1e6:.1f} MB - resuming ({attempt}/{ATTEMPTS})")
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:      # nothing left to download
                print()
                return True
            print(f"\n  server error {e.code} - retrying ({attempt}/{ATTEMPTS})")
        except (OSError, http.client.HTTPException) as e:
            print(f"\n  network problem ({e}) - retrying ({attempt}/{ATTEMPTS})")
        time.sleep(min(30, 3 * attempt))
    return False


def unpack(archive, target, expected_sha256):
    """Decompress a .bz2 file, verifying it is complete and matches the expected SHA-256."""
    part = target + '.part'
    decompressor, digest = bz2.BZ2Decompressor(), hashlib.sha256()
    try:
        with open(archive, 'rb') as src, open(part, 'wb') as out:
            for chunk in iter(lambda: src.read(CHUNK), b''):
                data = decompressor.decompress(chunk)
                out.write(data)
                digest.update(data)
    except (OSError, EOFError) as e:
        os.remove(part)
        return f'the downloaded archive is damaged ({e})'
    if not decompressor.eof:
        os.remove(part)
        return 'the downloaded archive is incomplete'
    if digest.hexdigest() != expected_sha256:
        os.remove(part)
        return 'checksum mismatch'
    os.replace(part, target)
    return None


def download(name, url, expected_sha256, target_dir=MODEL_DIR):
    target = os.path.join(target_dir, name)
    if os.path.isfile(target) and sha256_of(target) == expected_sha256:
        print(f"{name}: already downloaded")
        return True

    archive = target + '.bz2.part'
    print(f"{name}: downloading {url}")
    if not fetch(url, archive):
        print(f"{name}: download failed - check your internet connection and run this again "
              "(it continues where it stopped).")
        return False
    error = unpack(archive, target, expected_sha256)
    os.remove(archive)
    if error:
        print(f"{name}: {error} - please run this again.")
        return False
    print(f"{name}: OK")
    return True


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = [download(name, url, sha) for name, (url, sha) in MODELS.items()]
    if all(results):
        print(f"\nAll models are ready in {MODEL_DIR}")
        return 0
    print("\nSome models could not be downloaded.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
