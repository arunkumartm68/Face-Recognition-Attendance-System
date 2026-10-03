"""Tests for the face pipeline: feature extraction, recognition and attendance
marking, class sessions, accuracy evaluation and the registration window.

They need the dlib model files (python download_models.py) and are skipped
without them. The scanner-loop test opens an OpenCV window, so it only runs
with the environment variable RUN_GUI_TESTS=1.
"""
import _env  # noqa: F401  - must be imported before the app modules

import contextlib
import io
import json
import logging
import os
import time
import tkinter
import unittest
from datetime import datetime, timedelta
from unittest import mock

import cv2
import numpy as np

import attendance_taker
import common
import evaluate_accuracy
import features_extraction_to_csv
from face_utils import features_need_rebuild, imread_rgb, imwrite_rgb, load_features

MODELS_READY = not common.missing_model_files()
needs_models = unittest.skipUnless(MODELS_READY, 'dlib models not downloaded (python download_models.py)')
logging.disable(logging.WARNING)


def face_rgb():
    return imread_rgb(os.path.join(_env.FIXTURES, 'face.jpg'))


def variant(rgb, i):
    """Slightly different copies of the fixture, like successive webcam photos."""
    image = cv2.convertScaleAbs(rgb, alpha=1.0 + 0.05 * ((i % 3) - 1), beta=6 * ((i % 4) - 1.5))
    shift = np.float32([[1, 0, (i % 3) - 1], [0, 1, i % 2]])
    return cv2.warpAffine(image, shift, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def camera_frame_rgb(rgb=None):
    rgb = face_rgb() if rgb is None else rgb
    canvas = np.zeros((480, 640, 3), np.uint8)
    h, w = rgb.shape[:2]
    y, x = (480 - h) // 2, (640 - w) // 2
    canvas[y:y + h, x:x + w] = rgb
    return canvas


def camera_frame_bgr(rgb=None):
    return cv2.cvtColor(camera_frame_rgb(rgb), cv2.COLOR_RGB2BGR)


def add_person(number, reg, name, photos=4, rgb=None):
    folder = os.path.join(common.FACES_DIR, f'person_{number}_{reg}_{name}')
    os.makedirs(folder)
    for i in range(photos):
        assert imwrite_rgb(os.path.join(folder, f'img_face_{i + 1}.jpg'), variant(face_rgb() if rgb is None else rgb, i))
    return folder


def run_sql(sql, args=()):
    conn = common.connect()
    try:
        rows = conn.execute(sql, args).fetchall()
        conn.commit()
        return rows
    finally:
        conn.close()


def add_student(reg, name, password_hash='x'):
    run_sql('INSERT INTO students VALUES (?, ?, ?)', (reg, name, password_hash))


def build_features():
    with contextlib.redirect_stdout(io.StringIO()):
        features_extraction_to_csv.main()


class TestImageIO(unittest.TestCase):
    def setUp(self):
        _env.reset_state()

    def test_unicode_paths_roundtrip(self):
        folder = os.path.join(common.FACES_DIR, 'person_1_R1_José தமிழ்')
        os.makedirs(folder)
        path = os.path.join(folder, 'img_face_1.png')
        image = face_rgb()
        self.assertTrue(imwrite_rgb(path, image))
        np.testing.assert_array_equal(imread_rgb(path), image)

    def test_unreadable_files(self):
        self.assertIsNone(imread_rgb(os.path.join(common.FACES_DIR, 'missing.jpg')))
        empty = os.path.join(common.FACES_DIR, 'empty.jpg')
        open(empty, 'wb').close()
        self.assertIsNone(imread_rgb(empty))


class TestScannerArguments(unittest.TestCase):
    def test_parse_args(self):
        args = attendance_taker.parse_args(['--duration=60', '--late-after=7.5', '--started-by=Prof (-Maths)',
                                            '--', '-Maths'])
        self.assertEqual((args.subject, args.duration, args.late_after, args.started_by, args.source),
                         ('-Maths', 60.0, 7.5, 'Prof (-Maths)', '0'))
        self.assertEqual(attendance_taker.parse_args([]).subject, common.GENERAL_SUBJECT)


@needs_models
class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        _env.reset_state()

    def test_builds_one_profile_per_student(self):
        add_person(1, 'R1', 'Arun')
        add_person(2, 'R2', 'José K')
        build_features()
        labels, features = load_features()
        self.assertEqual(labels, ['R1_Arun', 'R2_José K'])
        self.assertEqual(features.shape, (2, 128))
        with open(common.FEATURES_META, encoding='utf-8') as f:
            self.assertEqual(json.load(f)['version'], common.FEATURES_VERSION)
        self.assertEqual([f for f in os.listdir(common.DATA_DIR) if f.endswith('.tmp')], [])
        self.assertFalse(features_need_rebuild())

    def test_detects_stale_features(self):
        folder = add_person(1, 'R1', 'Arun')
        self.assertTrue(features_need_rebuild())                 # no CSV yet
        build_features()
        self.assertFalse(features_need_rebuild())
        imwrite_rgb(os.path.join(folder, 'img_face_9.jpg'), face_rgb())
        self.assertTrue(features_need_rebuild())                 # new photo
        build_features()
        self.assertFalse(features_need_rebuild())
        for name in os.listdir(folder):
            os.remove(os.path.join(folder, name))
        os.rmdir(folder)
        self.assertTrue(features_need_rebuild())                 # deleted student
        build_features()
        with open(common.FEATURES_META, 'w', encoding='utf-8') as f:
            json.dump({'version': 1}, f)
        self.assertTrue(features_need_rebuild())                 # built by an older version

    def test_skips_photos_without_faces(self):
        add_person(1, 'R1', 'Arun')
        blank = os.path.join(common.FACES_DIR, 'person_2_R2_Blank')
        os.makedirs(blank)
        imwrite_rgb(os.path.join(blank, 'img_face_1.jpg'), np.full((200, 200, 3), 127, np.uint8))
        build_features()
        self.assertEqual(load_features()[0], ['R1_Arun'])


@needs_models
class TestRecognizer(unittest.TestCase):
    def setUp(self):
        _env.reset_state()
        add_student('R1', 'Arun')
        add_person(1, 'R1', 'Arun')
        build_features()

    def recognizer(self, subject=common.GENERAL_SUBJECT, **kwargs):
        recognizer = attendance_taker.Face_Recognizer(subject, **kwargs)
        recognizer.get_face_database()
        return recognizer

    def test_recognises_and_marks_present(self):
        recognizer = self.recognizer()
        recognizer.start_session()
        results = recognizer.process_frame(camera_frame_bgr())
        self.assertEqual(len(results), 1)
        label, dist, _rect = results[0]
        self.assertEqual(label, 'R1_Arun')
        self.assertLess(dist, common.MATCH_THRESHOLD)
        recognizer.process_frame(camera_frame_bgr())
        rows = run_sql('SELECT register_number, name, status, subject FROM attendance')
        self.assertEqual([tuple(r) for r in rows], [('R1', 'Arun', 'Present', common.GENERAL_SUBJECT)])
        recognizer.end_session()
        session = run_sql('SELECT * FROM sessions')[0]
        self.assertIsNotNone(session['ended_at'])
        self.assertEqual(session['subject'], common.GENERAL_SUBJECT)

    def test_marks_late_after_the_late_time(self):
        recognizer = self.recognizer(duration_min=60, late_after_min=10)
        start = datetime.now() - timedelta(minutes=15)
        recognizer.start_session(start)
        recognizer.process_frame(camera_frame_bgr())
        self.assertEqual(run_sql('SELECT status FROM attendance')[0][0], 'Late')
        session = run_sql('SELECT * FROM sessions')[0]
        self.assertEqual(session['planned_end'], (start + timedelta(minutes=60)).strftime(common.TIMESTAMP_FORMAT))
        self.assertEqual((session['duration_min'], session['late_after_min']), (60, 10))

    def test_no_match_above_threshold(self):
        recognizer = self.recognizer()
        recognizer.threshold = 0.01
        recognizer.start_session()
        label, dist, _rect = recognizer.process_frame(camera_frame_bgr())[0]
        self.assertIsNone(label)
        self.assertGreater(dist, 0.01)
        self.assertEqual(run_sql('SELECT COUNT(*) FROM attendance')[0][0], 0)
        self.assertEqual(recognizer.process_frame(np.zeros((480, 640, 3), np.uint8)), [])

    def test_attendance_rules(self):
        run_sql("INSERT INTO staff VALUES ('Maths', '', 'x')")
        add_student('R2', 'Priya')
        run_sql("INSERT INTO enrollments VALUES ('R2', 'Maths')")
        recognizer = self.recognizer('Maths')
        recognizer.start_session()
        self.assertEqual(recognizer.attendance('R9_Ghost'), 'not_registered')
        self.assertEqual(recognizer.attendance('R1_Arun'), 'not_enrolled')
        self.assertEqual(recognizer.attendance('R2_Priya'), 'marked')
        self.assertEqual(recognizer.attendance('R2_Priya'), 'skipped')
        self.assertEqual(self.recognizer('Maths').attendance('R2_Priya'), 'already')
        self.assertEqual([tuple(r) for r in run_sql('SELECT register_number, subject FROM attendance')],
                         [('R2', 'Maths')])

    def test_session_time_limit(self):
        timed = self.recognizer(duration_min=1)
        timed.start_session(datetime.now() - timedelta(minutes=2))
        self.assertTrue(timed.session_over(datetime.now()))
        untimed = self.recognizer()
        untimed.start_session(datetime.now() - timedelta(hours=5))
        self.assertFalse(untimed.session_over(datetime.now()))

    def test_camera_failure_creates_no_session(self):
        recognizer = self.recognizer()
        with mock.patch.object(attendance_taker, 'show_error') as show_error:
            self.assertEqual(recognizer.run(os.path.join(common.DATA_DIR, 'no-such-video.avi')), 1)
        show_error.assert_called_once()
        self.assertEqual(run_sql('SELECT COUNT(*) FROM sessions')[0][0], 0)

    @unittest.skipUnless(os.environ.get('RUN_GUI_TESTS') == '1', 'opens an OpenCV window (set RUN_GUI_TESTS=1)')
    def test_scanner_loop_with_video_file(self):
        path = os.path.join(common.DATA_DIR, 'clip.avi')

        def write_clip(frames):
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
            for _ in range(frames):
                writer.write(camera_frame_bgr())
            writer.release()

        write_clip(15)          # whole clip is processed, then the stream ends
        self.assertEqual(self.recognizer().run(path), 0)
        self.assertEqual(run_sql('SELECT COUNT(*) FROM attendance')[0][0], 1)
        self.assertIsNotNone(run_sql('SELECT ended_at FROM sessions')[0][0])

        write_clip(600)         # ~60 s of video, but the class is only 3 seconds long
        recognizer = self.recognizer(duration_min=0.05)
        started = time.time()
        self.assertEqual(recognizer.run(path), 0)
        self.assertLess(time.time() - started, 30)
        self.assertLess(recognizer.frame_cnt, 600)
        self.assertEqual(run_sql('SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL')[0][0], 2)


@needs_models
class TestEvaluation(unittest.TestCase):
    def setUp(self):
        _env.reset_state()

    def evaluate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = evaluate_accuracy.run_evaluation()
        with open(common.EVALUATION_JSON, encoding='utf-8') as f:
            return code, json.load(f)

    def test_scoring(self):
        genuine = [(0.30, 0.80), (0.50, 0.90), (0.40, 0.35)]
        impostor = [0.30, 0.70]
        row = evaluate_accuracy.score(genuine, impostor, 0.45)
        self.assertEqual((row['correct_matches'], row['false_rejects'], row['wrong_person']), (1, 1, 1))
        self.assertEqual((row['false_accepts'], row['true_rejects']), (1, 1))
        self.assertEqual(row['accuracy'], 40.0)
        rows = [evaluate_accuracy.score(genuine, impostor, th) for th in (0.2, 0.45, 0.6)]
        self.assertIn(evaluate_accuracy.recommend(rows), (0.2, 0.45, 0.6))
        self.assertIsNone(evaluate_accuracy.recommend([evaluate_accuracy.score(genuine, [], 0.45)]))

    def test_no_photos(self):
        code, result = self.evaluate()
        self.assertEqual(code, 1)
        self.assertIn('register students first', result['error'])

    def test_single_student(self):
        add_person(1, 'R1', 'Arun')
        code, result = self.evaluate()
        self.assertEqual(code, 0)
        self.assertEqual(result['dataset']['students'], 1)
        self.assertEqual(result['dataset']['impostor_trials'], 0)
        self.assertIsNone(result['recommended_threshold'])
        self.assertTrue(any('Only one student' in note for note in result['notes']))
        row = next(r for r in result['thresholds'] if abs(r['threshold'] - common.MATCH_THRESHOLD) < 1e-9)
        self.assertEqual(row['tar'], 100.0)
        self.assertIsNone(row['far'])

    def test_two_students(self):
        add_person(1, 'R1', 'Arun')
        add_person(2, 'R2', 'Mirror', rgb=cv2.flip(face_rgb(), 1))
        code, result = self.evaluate()
        self.assertEqual(code, 0)
        self.assertEqual((result['dataset']['genuine_trials'], result['dataset']['impostor_trials']), (8, 8))
        self.assertIsNotNone(result['recommended_threshold'])
        for key in ('color_conversion', 'detection', 'landmarks', 'descriptor', 'matching', 'sqlite_write'):
            self.assertGreater(result['latency_ms'][key], 0)
        self.assertAlmostEqual(result['latency_ms']['fps_estimate'],
                               1000 / result['latency_ms']['per_frame_total'], delta=0.1)


def _tk_available():
    try:
        root = tkinter.Tk()
        root.destroy()
        return True
    except tkinter.TclError:
        return False


@needs_models
@unittest.skipUnless(_tk_available(), 'needs a display for Tkinter')
class TestRegistrationWindow(unittest.TestCase):
    def setUp(self):
        _env.reset_state()
        import get_faces_from_camera_tkinter as register
        self.module = register
        self.window = register.Face_Register()
        self.window.win.withdraw()
        self.window.pre_work_mkdir()
        self.window.check_existing_faces_cnt()
        self.window.GUI_info()
        self.addCleanup(self.window.win.destroy)

    def enter(self, reg, name, password):
        for entry, value in ((self.window.input_reg, reg), (self.window.input_name, name),
                             (self.window.input_pwd, password)):
            entry.delete(0, 'end')
            entry.insert(0, value)
        self.window.GUI_get_input_name()
        return self.window.log_all['text']

    def test_validates_details(self):
        self.assertIn("'_'", self.enter('21_CS_001', 'Arun', 'secret1'))
        self.assertIn('at least', self.enter('21CS001', 'Arun', '123'))
        self.assertIn('password', self.enter('21CS001', 'Arun', ''))
        self.assertEqual(run_sql('SELECT COUNT(*) FROM students')[0][0], 0)
        self.assertEqual(os.listdir(common.FACES_DIR), [])

    def test_registers_student_and_keeps_existing_password(self):
        self.assertIn('created', self.enter('21CS001', 'Arun', 'secret1'))
        row = run_sql('SELECT * FROM students')[0]
        self.assertTrue(common.verify_password(row['password'], 'secret1'))
        self.assertTrue(os.path.isdir(os.path.join(common.FACES_DIR, 'person_1_21CS001_Arun')))
        self.assertIn('Already registered', self.enter('21CS001', 'Arun', 'secret1'))

        add_student('21CS002', 'Priya', common.hash_password('priya123'))
        self.assertIn('existing password kept', self.enter('21CS002', 'Priya S', ''))
        row = run_sql("SELECT * FROM students WHERE register_number = '21CS002'")[0]
        self.assertEqual(row['name'], 'Priya S')
        self.assertTrue(common.verify_password(row['password'], 'priya123'))
        self.assertTrue(os.path.isdir(os.path.join(common.FACES_DIR, 'person_2_21CS002_Priya S')))

    def test_saves_clean_photos_and_starts_extraction(self):
        self.enter('21CS001', 'Arun', 'secret1')
        frame = camera_frame_rgb()
        self.window.process_frame(frame.copy())
        self.assertEqual(self.window.current_frame_faces_cnt, 1)
        self.assertFalse(np.array_equal(self.window.current_frame, frame))      # HUD drawn on the preview
        np.testing.assert_array_equal(self.window.current_frame_clean, frame)   # ...but not on the clean copy

        with mock.patch.object(self.module.subprocess, 'Popen') as popen:
            self.window.save_current_face()
            self.assertIsNotNone(self.window.reference_desc)
            saved = imread_rgb(os.path.join(self.window.current_face_dir, 'img_face_1.jpg'))
            w = self.window
            expected = frame[max(0, w.face_ROI_height_start - w.hh):w.face_ROI_height_start + w.face_ROI_height + w.hh,
                             max(0, w.face_ROI_width_start - w.ww):w.face_ROI_width_start + w.face_ROI_width + w.ww]
            self.assertEqual(saved.shape, expected.shape)
            self.assertLess(np.abs(saved.astype(int) - expected.astype(int)).mean(), 3)   # JPEG noise only

            self.window.process_frame(frame.copy())
            self.assertLess(self.window.live_distance, common.MATCH_THRESHOLD)

            self.window.current_frame_faces_cnt = 2
            self.window.save_current_face()
            self.assertIn('Multiple faces', self.window.log_all['text'])
            self.window.current_frame_faces_cnt = 1

            for _ in range(common.PHOTOS_PER_STUDENT - 1):
                self.window.save_current_face()
            popen.assert_called_once()
            self.assertEqual(popen.call_args.args[0][1], 'features_extraction_to_csv.py')

        photos = sorted(os.listdir(self.window.current_face_dir))
        self.assertEqual(len(photos), common.PHOTOS_PER_STUDENT)
        with open(common.LATENCY_JSON, encoding='utf-8') as f:
            latency = json.load(f)
        self.assertEqual(latency['photos_count'], common.PHOTOS_PER_STUDENT)
        self.assertTrue(all(ms > 0 for ms in latency['latencies']))

    def test_clear_keeps_hidden_files_and_needs_confirmation(self):
        self.enter('21CS001', 'Arun', 'secret1')
        open(os.path.join(common.FACES_DIR, '.gitkeep'), 'w').close()
        with mock.patch.object(self.module.messagebox, 'askyesno', return_value=False):
            self.window.GUI_clear_data()
        self.assertEqual(len(os.listdir(common.FACES_DIR)), 2)
        with mock.patch.object(self.module.messagebox, 'askyesno', return_value=True):
            self.window.GUI_clear_data()
        self.assertEqual(os.listdir(common.FACES_DIR), ['.gitkeep'])


if __name__ == '__main__':
    unittest.main()
