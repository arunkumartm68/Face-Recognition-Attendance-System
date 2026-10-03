"""
Evaluation Script: real recognition accuracy and per-stage latency, measured on
the registered face photos in data/data_faces_from_camera/.

* Genuine trials (leave-one-out): every photo is matched against all student
  profiles, with that photo left out of its owner's averaged profile - like a
  registered student walking up to the scanner.
* Impostor trials: every photo is matched with its owner's profile removed, so
  any match is a false accept (needs at least two registered students).

Results are printed and saved to data/evaluation_results.json, which the admin
dashboard and the server start-up banner display.
"""

import json
import os
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

import common
from face_utils import detect_faces, imread_rgb, largest_face, load_models

THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
FRAME_SIZE = (640, 480)   # webcam frames are processed at this size


def as_camera_frame(rgb):
    """Place a registration photo on a 640x480 canvas so detection is timed on a full frame."""
    width, height = FRAME_SIZE
    h, w = rgb.shape[:2]
    scale = min(width / w, height / h, 1.0)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
        h, w = rgb.shape[:2]
    canvas = np.zeros((height, width, 3), np.uint8)
    y, x = (height - h) // 2, (width - w) // 2
    canvas[y:y + h, x:x + w] = rgb
    return canvas


def collect_descriptors(models):
    """[(register_number, descriptor)], per-stage timings in ms, number of photos found."""
    detector, predictor, model = models
    samples, timing, photos = [], defaultdict(list), 0
    if not os.path.isdir(common.FACES_DIR):
        return samples, timing, photos
    for folder in sorted(os.listdir(common.FACES_DIR)):
        path = os.path.join(common.FACES_DIR, folder)
        if not os.path.isdir(path):
            continue
        # Identity = register number, so a re-registered student is one person.
        person = common.split_label(common.folder_label(folder))[0]
        for photo in sorted(os.listdir(path)):
            if not photo.lower().endswith(common.IMAGE_EXTENSIONS):
                continue
            photos += 1
            rgb = imread_rgb(os.path.join(path, photo))
            if rgb is None:
                continue
            frame_bgr = cv2.cvtColor(as_camera_frame(rgb), cv2.COLOR_RGB2BGR)

            t0 = time.perf_counter()
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t1 = time.perf_counter()
            faces = detect_faces(detector, frame)
            t2 = time.perf_counter()
            if not faces:
                continue
            shape = predictor(frame, largest_face(faces))
            t3 = time.perf_counter()
            descriptor = np.array(model.compute_face_descriptor(frame, shape), dtype=float)
            t4 = time.perf_counter()

            timing['color_conversion'].append((t1 - t0) * 1000)
            timing['detection'].append((t2 - t1) * 1000)
            timing['landmarks'].append((t3 - t2) * 1000)
            timing['descriptor'].append((t4 - t3) * 1000)
            samples.append((person, descriptor))
    return samples, timing, photos


def run_trials(samples):
    """Distances for scoring every threshold.

    genuine:  [(distance to own leave-one-out profile, distance to nearest other profile)]
    impostor: [distance to nearest other profile]
    """
    by_person = defaultdict(list)
    for person, descriptor in samples:
        by_person[person].append(descriptor)
    sums = {p: np.sum(v, axis=0) for p, v in by_person.items()}
    counts = {p: len(v) for p, v in by_person.items()}
    persons = sorted(by_person)
    profiles = {p: sums[p] / counts[p] for p in persons}

    genuine, impostor = [], []
    for person, descriptor in samples:
        others = [profiles[p] for p in persons if p != person]
        nearest_other = (float(np.min(np.linalg.norm(np.array(others) - descriptor, axis=1)))
                         if others else float('inf'))
        if others:
            impostor.append(nearest_other)
        if counts[person] >= 2:
            own_profile = (sums[person] - descriptor) / (counts[person] - 1)
            genuine.append((float(np.linalg.norm(own_profile - descriptor)), nearest_other))
    single_photo = sum(1 for p in persons if counts[p] < 2)
    return genuine, impostor, len(persons), single_photo, np.array([profiles[p] for p in persons])


def score(genuine, impostor, threshold):
    correct = false_rejects = wrong_person = 0
    for own, other in genuine:
        if min(own, other) >= threshold:
            false_rejects += 1
        elif own <= other:
            correct += 1
        else:
            wrong_person += 1
    false_accepts = sum(1 for d in impostor if d < threshold)
    true_rejects = len(impostor) - false_accepts

    def pct(part, whole):
        return round(100.0 * part / whole, 2) if whole else None

    return {
        'threshold': threshold,
        'genuine_trials': len(genuine), 'correct_matches': correct,
        'false_rejects': false_rejects, 'wrong_person': wrong_person,
        'impostor_trials': len(impostor), 'false_accepts': false_accepts, 'true_rejects': true_rejects,
        'tar': pct(correct, len(genuine)), 'frr': pct(false_rejects, len(genuine)),
        'wrong_rate': pct(wrong_person, len(genuine)), 'far': pct(false_accepts, len(impostor)),
        'accuracy': pct(correct + true_rejects, len(genuine) + len(impostor)),
    }


def recommend(rows):
    """Threshold with the best accuracy (ties: fewest wrong matches, then the stricter one)."""
    usable = [r for r in rows if r['genuine_trials'] and r['impostor_trials']]
    if not usable:
        return None
    best = max(usable, key=lambda r: (r['accuracy'], -(r['false_accepts'] + r['wrong_person']), -r['threshold']))
    return best['threshold']


def time_matching(samples, profiles):
    times = []
    for _, descriptor in samples:
        t0 = time.perf_counter()
        int(np.argmin(np.linalg.norm(profiles - descriptor, axis=1)))
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def time_sqlite_write(repeats=30):
    """Average time of one attendance write as the scanner does it: connect, insert, commit, close."""
    times = []
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, 'benchmark.db')
        conn = sqlite3.connect(db)
        conn.execute(common.ATTENDANCE_DDL)
        conn.close()
        for i in range(repeats):
            t0 = time.perf_counter()
            conn = sqlite3.connect(db)
            conn.execute('INSERT OR IGNORE INTO attendance (register_number, name, time, date, subject, status) '
                         'VALUES (?, ?, ?, ?, ?, ?)', (f'R{i}', 'Benchmark', '09:00:00 AM', '2000-01-01',
                                                       'Benchmark', common.STATUS_PRESENT))
            conn.commit()
            conn.close()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def save_results(result):
    os.makedirs(common.DATA_DIR, exist_ok=True)
    tmp = f'{common.EVALUATION_JSON}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, common.EVALUATION_JSON)


def fmt(value, suffix='%'):
    return 'n/a' if value is None else f'{value:.2f}{suffix}'


def print_report(result):
    ds = result['dataset']
    print("\n" + "=" * 100)
    print(" RECOGNITION ACCURACY ACROSS DISTANCE THRESHOLDS (measured)")
    print(f" {ds['students']} students, {ds['faces_detected']}/{ds['photos']} photos with a detected face, "
          f"{ds['genuine_trials']} genuine and {ds['impostor_trials']} impostor trials")
    print("=" * 100)
    print(f" {'Threshold':<20} | {'Correct (TAR)':<14} | {'False reject':<13} | {'Wrong person':<13} | "
          f"{'False accept':<13} | {'Accuracy':<9}")
    print("-" * 100)
    for row in result['thresholds']:
        tags = []
        if abs(row['threshold'] - result['configured_threshold']) < 1e-9:
            tags.append('in use')
        if result['recommended_threshold'] is not None and abs(row['threshold'] - result['recommended_threshold']) < 1e-9:
            tags.append('best')
        label = f"{row['threshold']:.2f}" + (f" ({', '.join(tags)})" if tags else '')
        print(f" {label:<20} | {fmt(row['tar']):<14} | {fmt(row['frr']):<13} | {fmt(row['wrong_rate']):<13} | "
              f"{fmt(row['far']):<13} | {fmt(row['accuracy']):<9}")
    print("-" * 100)

    lat = result['latency_ms']
    print("\n" + "=" * 100)
    print(" PROCESSING TIME PER STAGE (measured on 640x480 frames, one face)")
    print("=" * 100)
    for key, label in (('color_conversion', 'BGR-to-RGB conversion'),
                       ('detection', 'Dlib HOG face detection'),
                       ('landmarks', '68-landmark localisation'),
                       ('descriptor', 'ResNet 128D descriptor'),
                       ('matching', 'Distance matching against all profiles')):
        print(f" {label:<48} | {lat[key]:.2f} ms")
    print("-" * 100)
    print(f" {'Total per frame':<48} | {lat['per_frame_total']:.2f} ms (~{lat['fps_estimate']:.1f} FPS)")
    print(f" {'SQLite attendance write (once per student)':<48} | {lat['sqlite_write']:.2f} ms")
    print("=" * 100)
    for note in result['notes']:
        print(" * " + note)
    print(f"\nSaved to {common.EVALUATION_JSON}\n")


def run_evaluation():
    print("Loading Dlib Models...")
    models = load_models()
    print(f"Extracting face descriptors from {common.FACES_DIR} ...")
    samples, timing, photos = collect_descriptors(models)
    result = {'evaluated_at': datetime.now().strftime(common.TIMESTAMP_FORMAT),
              'configured_threshold': common.MATCH_THRESHOLD}
    if not samples:
        result['error'] = ('No registered face photos found - register students first.' if photos == 0
                           else f'No face could be detected in any of the {photos} photos.')
        save_results(result)
        print(result['error'])
        return 1

    genuine, impostor, students, single_photo, profiles = run_trials(samples)
    thresholds = sorted(set(THRESHOLDS) | {round(common.MATCH_THRESHOLD, 2)})
    rows = [score(genuine, impostor, th) for th in thresholds]

    latency = {key: round(float(np.mean(values)), 2) for key, values in timing.items()}
    latency['matching'] = round(time_matching(samples, profiles), 3)
    latency['per_frame_total'] = round(sum(latency[k] for k in
                                           ('color_conversion', 'detection', 'landmarks', 'descriptor', 'matching')), 2)
    latency['fps_estimate'] = round(1000.0 / latency['per_frame_total'], 1)
    latency['sqlite_write'] = round(time_sqlite_write(), 2)

    notes = ["Measured on the registration photos: this shows how well registered students are told apart. "
             "Live conditions (lighting, distance, motion) can be harder."]
    if students < 2:
        notes.append("Only one student is registered, so false accepts cannot be measured yet "
                     "and no threshold is recommended.")
    if single_photo:
        notes.append(f"{single_photo} student(s) have a single photo and are left out of the genuine trials.")
    if not genuine:
        notes.append("Every student has only one photo - save more photos per student for genuine trials.")

    result.update({
        'dataset': {'students': students, 'photos': photos, 'faces_detected': len(samples),
                    'genuine_trials': len(genuine), 'impostor_trials': len(impostor)},
        'thresholds': rows,
        'recommended_threshold': recommend(rows),
        'latency_ms': latency,
        'notes': notes,
    })
    save_results(result)
    print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
