"""Live attendance scanner.

Recognises registered faces from the webcam and records attendance for one
subject. Every run is a class session: students recognised after the "late
after" time are marked Late, and the scanner stops by itself when the class
time is over.

    python attendance_taker.py "Maths" --duration 60 --late-after 10
"""
import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

import common
from common import (DATE_FORMAT, GENERAL_SUBJECT, MATCH_THRESHOLD, STATUS_LATE, STATUS_PRESENT,
                    TIME_FORMAT, TIMESTAMP_FORMAT, connect, init_db, on_roster, split_label)
from face_utils import (detect_faces, face_descriptor, features_need_rebuild, load_features,
                        load_models, nearest)

WINDOW = "camera"


def _mmss(delta):
    seconds = max(0, int(delta.total_seconds()))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class Face_Recognizer:
    def __init__(self, subject=GENERAL_SUBJECT, duration_min=0.0, late_after_min=0.0,
                 started_by='command line', threshold=MATCH_THRESHOLD):
        self.subject = subject
        self.duration_min = max(0.0, float(duration_min))
        self.late_after_min = max(0.0, float(late_after_min))
        self.started_by = started_by
        self.threshold = threshold
        self.models = load_models()
        self.font = cv2.FONT_ITALIC

        # Known faces from features_all.csv
        self.face_name_known_list = []
        self.face_features_known_list = np.zeros((0, 128))

        # Class session
        self.session_id = None
        self.session_start = None
        self.planned_end = None

        # (label, date) pairs already written to the DB this run. Without this
        # the loop would query SQLite once per face per frame.
        self.marked_this_session = set()

        # Telemetry
        self.frame_cnt = 0
        self.frame_ms = 0.0
        self.fps = 0.0
        self._last_tick = None

    def get_face_database(self):
        """Load known faces, rebuilding features_all.csv first if it is stale."""
        if features_need_rebuild():
            logging.info("Updating 128D feature database (features_all.csv)...")
            subprocess.run([sys.executable, 'features_extraction_to_csv.py'], cwd=common.BASE_DIR)
        self.face_name_known_list, self.face_features_known_list = load_features()
        if self.face_name_known_list:
            logging.info("Loaded %d registered face profiles from database.", len(self.face_name_known_list))
        else:
            logging.warning("No registered faces found in 'features_all.csv' - register students first.")
        return len(self.face_name_known_list)

    # ------------------------------------------------------------ session

    def start_session(self, now=None):
        now = now or datetime.now()
        self.session_start = now
        self.planned_end = now + timedelta(minutes=self.duration_min) if self.duration_min else None
        conn = connect()
        try:
            cur = conn.execute(
                'INSERT INTO sessions (subject, date, started_at, planned_end, duration_min, '
                'late_after_min, started_by) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (self.subject, now.strftime(DATE_FORMAT), now.strftime(TIMESTAMP_FORMAT),
                 self.planned_end.strftime(TIMESTAMP_FORMAT) if self.planned_end else None,
                 self.duration_min, self.late_after_min, self.started_by))
            conn.commit()
            self.session_id = cur.lastrowid
        finally:
            conn.close()

    def end_session(self, now=None):
        if self.session_id is None:
            return
        now = now or datetime.now()
        conn = connect()
        try:
            conn.execute('UPDATE sessions SET ended_at = ? WHERE id = ?',
                         (now.strftime(TIMESTAMP_FORMAT), self.session_id))
            conn.commit()
        finally:
            conn.close()
        self.session_id = None

    def status_at(self, now):
        if (self.late_after_min and self.session_start is not None
                and now - self.session_start >= timedelta(minutes=self.late_after_min)):
            return STATUS_LATE
        return STATUS_PRESENT

    def session_over(self, now):
        return self.planned_end is not None and now >= self.planned_end

    # ------------------------------------------------------------ marking

    def attendance(self, label, now=None):
        """Record attendance for a profile label "<reg>_<name>".

        Returns 'marked', 'already', 'not_registered', 'not_enrolled' or
        'skipped' (already handled earlier in this run).
        """
        now = now or datetime.now()
        current_date = now.strftime(DATE_FORMAT)
        if (label, current_date) in self.marked_this_session:
            return 'skipped'

        register_number, name = split_label(label)
        conn = connect()
        try:
            # Deleting a student in the dashboard removes them from the DB; an
            # old face profile must not keep creating attendance rows.
            student = conn.execute('SELECT name FROM students WHERE register_number = ?',
                                   (register_number,)).fetchone()
            if student is None:
                outcome = 'not_registered'
                print(f"{label} is not a registered student - attendance not recorded")
            elif not on_roster(conn, register_number, self.subject):
                outcome = 'not_enrolled'
                print(f"{label} is not assigned to {self.subject} - attendance not recorded")
            else:
                status = self.status_at(now)
                current_time = now.strftime(TIME_FORMAT)
                cur = conn.execute(
                    'INSERT OR IGNORE INTO attendance (register_number, name, time, date, subject, status) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (register_number, student['name'] or name, current_time, current_date,
                     self.subject, status))
                conn.commit()
                if cur.rowcount:
                    outcome = 'marked'
                    print(f"{label} marked {status} for {current_date} on {self.subject} at {current_time}")
                else:
                    outcome = 'already'
                    print(f"{label} is already marked for {current_date} on {self.subject}")
        finally:
            conn.close()

        self.marked_this_session.add((label, current_date))
        return outcome

    # ------------------------------------------------------------ frames

    def process_frame(self, img_bgr, now=None):
        """Detect, recognise and mark every face in a BGR camera frame.

        Returns [(label or None, distance or None, dlib rectangle), ...].
        """
        detector, predictor, model = self.models
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)     # dlib's models expect RGB
        results = []
        for rect in detect_faces(detector, rgb):
            descriptor = face_descriptor(predictor, model, rgb, rect)
            index, dist = nearest(descriptor, self.face_features_known_list)
            label = None
            if index is not None and dist < self.threshold:
                label = self.face_name_known_list[index]
                self.attendance(label, now)
            logging.debug("Face: %s (distance %s)", label or 'unknown', dist)
            results.append((label, dist, rect))
        return results

    def _tick(self):
        now = time.perf_counter()
        if self._last_tick is not None and now > self._last_tick:
            self.fps = 1.0 / (now - self._last_tick)
        self._last_tick = now

    def draw_note(self, img, results, now):
        """Telemetry and session info, plus a box and label for every face."""
        def put(text, y, color=(0, 255, 0), scale=0.6):
            cv2.putText(img, text, (20, y), self.font, scale, color, 1, cv2.LINE_AA)

        put("Face Recognizer (Dlib ResNet)", 30, (255, 255, 255), 0.8)
        put(f"Subject: {self.subject}", 58, (255, 255, 255))
        put(f"Processing: {self.frame_ms:.1f} ms  ({self.fps:.1f} FPS)", 84)
        put(f"Frame: {self.frame_cnt}  |  Faces: {len(results)}  |  Match if Dist < {self.threshold:.2f}", 108)
        if self.session_start is not None:
            line = f"Session: {_mmss(now - self.session_start)} elapsed"
            if self.planned_end is not None:
                line += f", {_mmss(self.planned_end - now)} left"
            put(line, 132, (255, 255, 0))
            if self.late_after_min:
                mode = "late" if self.status_at(now) == STATUS_LATE else "on time"
                put(f"Arrivals now count as {mode} (late after {self.late_after_min:g} min)", 156, (255, 255, 0))
        put("Press 'Q' to quit", img.shape[0] - 20, (255, 255, 255))

        for label, dist, rect in results:
            color = (0, 255, 0) if label else (0, 0, 255)
            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), color, 2)
            dist_str = f"Dist: {dist:.3f}" if dist is not None else "no registered faces"
            text = f"{label} ({dist_str})" if label else f"Unknown ({dist_str})"
            cv2.putText(img, text, (rect.left(), max(rect.top() - 10, 15)), self.font, 0.6, color, 1, cv2.LINE_AA)

    def run(self, source=0):
        """Scan until 'Q', the window is closed, the class time is over or the video ends."""
        self.get_face_database()
        cap = open_capture(source)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok or frame is None:
            cap.release()
            show_error("Could not open the camera.",
                       "Check it is connected and not used by another app.")
            return 1

        self.start_session()
        print(f"Attendance session started for {self.subject}"
              + (f", stops automatically after {self.duration_min:g} min" if self.duration_min else "")
              + (f", late after {self.late_after_min:g} min" if self.late_after_min else ""))
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        try:
            while ok and frame is not None:
                now = datetime.now()
                if self.session_over(now):
                    print("Class time is over - the scanner stopped automatically.")
                    break
                self.frame_cnt += 1
                t0 = time.perf_counter()
                results = self.process_frame(frame, now)
                self.frame_ms = (time.perf_counter() - t0) * 1000
                self._tick()
                self.draw_note(frame, results, now)
                cv2.imshow(WINDOW, frame)

                kk = cv2.waitKey(1) & 0xFF
                if kk in (ord('q'), ord('Q')):
                    break
                # Window closed with the X button - otherwise imshow() would
                # immediately re-create it and only Q could stop the scanner.
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                ok, frame = cap.read()
            else:
                logging.info("Video stream ended.")
        finally:
            self.end_session()
            cap.release()
            cv2.destroyAllWindows()
        return 0


def open_capture(source):
    if isinstance(source, int) and sys.platform == 'win32':
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(source)


def show_error(*lines, seconds=6):
    """Print an error and show it in a window, since the scanner is usually started from the browser."""
    print(" ".join(lines), file=sys.stderr)
    canvas = np.zeros((60 + 40 * len(lines), 720, 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (20, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(WINDOW, canvas)
    cv2.waitKey(seconds * 1000)
    cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Face recognition attendance scanner")
    parser.add_argument('subject', nargs='?', default=GENERAL_SUBJECT,
                        help="subject to record attendance for (default: General)")
    parser.add_argument('--duration', type=float, default=0.0,
                        help="class length in minutes; the scanner stops automatically afterwards (0 = no limit)")
    parser.add_argument('--late-after', type=float, default=0.0,
                        help="minutes after the start from which arrivals are marked Late (0 = never)")
    parser.add_argument('--source', default='0',
                        help="camera index, video file or stream URL (default: 0, the first webcam)")
    parser.add_argument('--started-by', default='command line', help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    init_db()
    source = int(args.source) if args.source.isdigit() else args.source
    recognizer = Face_Recognizer(args.subject, args.duration, args.late_after, args.started_by)
    return recognizer.run(source)


if __name__ == '__main__':
    sys.exit(main())
