"""Face registration window: enter a student's details, then save face photos from the webcam."""
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import tkinter as tk
from tkinter import font as tkFont
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import common
from common import (MATCH_THRESHOLD, PHOTOS_PER_STUDENT, connect, hash_password, init_db,
                    validate_name, validate_password, validate_register_number)
from face_utils import detect_faces, face_descriptor, imwrite_rgb, load_models

_PERSON_NUMBER_RE = re.compile(r'^person_(\d+)_')


class Face_Register:

    def __init__(self):
        self.detector, self.predictor, self.face_reco_model = load_models()

        self.current_frame_faces_cnt = 0  # cnt for counting faces in current frame
        self.existing_faces_cnt = 0  # highest person_<n> number used so far
        self.ss_cnt = 0  # photos saved for the current student

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Face Register")

        # PLease modify window size here if needed
        self.win.geometry("1000x500")

        # GUI left part
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # GUI right part
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text="0")
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_reg = tk.Entry(self.frame_right_info)
        self.input_pwd = tk.Entry(self.frame_right_info, show="*")
        self.input_name_char = ""
        # Guards against a second "Input" click creating a duplicate profile.
        self.registered_key = None
        self.extraction_launched = False
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info, wraplength=340, justify=tk.LEFT)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = common.FACES_DIR
        self.current_face_dir = ""

        # Current frame and face ROI position
        self.current_frame = None
        # Untouched copy of the frame. current_frame gets the HUD text and
        # bounding box drawn onto it, which must not end up in saved photos.
        self.current_frame_clean = None
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        # Live consistency check: distance between the face in front of the
        # camera and this student's first saved photo.
        self.current_desc = None
        self.reference_desc = None
        self.live_distance = None

        self.face_folder_created_flag = False
        self.capture_latencies = []

        # FPS / timing
        self.process_ms = 0.0
        self.fps = 0.0
        self._last_tick = None

        self.cap = None  # opened in run()

    #  Delete old face folders
    def GUI_clear_data(self):
        # This wipes every student's photos, not just the one being registered.
        if not messagebox.askyesno(
                "Clear all face data",
                "This deletes the saved face photos of EVERY registered student "
                "and features_all.csv.\n\nContinue?"):
            return
        for entry in os.listdir(self.path_photos_from_camera):
            if entry.startswith('.'):   # keep .gitkeep and other hidden files
                continue
            target = os.path.join(self.path_photos_from_camera, entry)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                try:
                    os.remove(target)
                except OSError as e:
                    logging.warning("Could not remove %s: %s", target, e)
        for path in (common.FEATURES_CSV, common.FEATURES_META):
            if os.path.isfile(path):
                os.remove(path)
        self.registered_key = None
        self.face_folder_created_flag = False
        self.reference_desc = None
        self.ss_cnt = 0
        self.existing_faces_cnt = 0
        self.label_cnt_face_in_database['text'] = "0"
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_get_input_name(self):
        reg = self.input_reg.get().strip()
        name = self.input_name.get().strip()
        password = self.input_pwd.get()

        error = validate_register_number(reg) or validate_name(name)
        if error:
            self.log_all["text"] = error
            return
        try:
            conn = connect()
            try:
                existing = conn.execute('SELECT 1 FROM students WHERE register_number = ?',
                                        (reg,)).fetchone() is not None
            finally:
                conn.close()
        except sqlite3.Error as e:
            self.log_all["text"] = f"Database error: {e}"
            return
        if password:
            error = validate_password(password)
        elif not existing:
            # A new student needs a password to log in to the web dashboard.
            error = "Please set a login password for this student!"
        if error:
            self.log_all["text"] = error
            return

        key = f"{reg}_{name}"
        if self.registered_key == key and self.face_folder_created_flag:
            # Second click on "Input" with unchanged details: keep the folder
            # already created instead of starting a duplicate profile.
            self.log_all["text"] = f"Already registered as \"{self.current_face_dir}\" - keep saving faces."
            return

        try:
            conn = connect()
            try:
                if password:
                    conn.execute(
                        'INSERT INTO students (register_number, name, password) VALUES (?, ?, ?) '
                        'ON CONFLICT(register_number) DO UPDATE SET name=excluded.name, password=excluded.password',
                        (reg, name, hash_password(password)))
                else:
                    conn.execute('UPDATE students SET name = ? WHERE register_number = ?', (name, reg))
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as e:
            self.log_all["text"] = f"Database error: {e}"
            return

        self.input_name_char = key
        self.capture_latencies = []
        self.extraction_launched = False
        self.reference_desc = None
        if os.path.exists(common.LATENCY_JSON):
            try:
                os.remove(common.LATENCY_JSON)
            except OSError:
                pass

        try:
            self.create_face_folder()
        except OSError as e:
            self.log_all["text"] = f"Could not create folder: {e}"
            return

        self.registered_key = key
        self.label_cnt_face_in_database['text'] = str(self.count_registered_folders())
        if existing and not password:
            self.log_all["text"] += " (existing password kept)"

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                 text="Face register",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="Faces in current frame: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 1: Clear face photos").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info,
                  text='Clear',
                  command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 2: Input details
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 2: Input Details").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.frame_right_info, text="Reg No: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_reg.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Name: ").grid(row=9, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=9, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Password: ").grid(row=10, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_pwd.grid(row=10, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.frame_right_info,
                  text='Input',
                  command=self.GUI_get_input_name).grid(row=10, column=2, padx=5)

        tk.Label(self.frame_right_info, fg="gray",
                 text="(leave Password blank to keep an existing student's password)"
                 ).grid(row=11, column=0, columnspan=3, sticky=tk.W, padx=5)

        # Step 3: Save current face in frame
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text=f"Step 3: Save face image ({PHOTOS_PER_STUDENT} photos)").grid(
            row=12, column=0, columnspan=3, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.frame_right_info,
                  text='Save current face',
                  command=self.save_current_face).grid(row=13, column=0, columnspan=3, sticky=tk.W)

        # Show log in GUI
        self.log_all.grid(row=14, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack()

    # Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        os.makedirs(self.path_photos_from_camera, exist_ok=True)

    def _person_numbers(self):
        numbers = []
        for person in os.listdir(self.path_photos_from_camera):
            match = _PERSON_NUMBER_RE.match(person)
            if match and os.path.isdir(os.path.join(self.path_photos_from_camera, person)):
                numbers.append(int(match.group(1)))
        return numbers

    def count_registered_folders(self):
        return len(self._person_numbers())

    # Start from person_x+1
    def check_existing_faces_cnt(self):
        # Ignores stray files and any folder without a numeric "person_<n>_" prefix.
        numbers = self._person_numbers()
        self.existing_faces_cnt = max(numbers) if numbers else 0
        self.label_cnt_face_in_database['text'] = str(len(numbers))

    def update_fps(self):
        now = time.perf_counter()
        if self._last_tick is not None and now > self._last_tick:
            self.fps = 1.0 / (now - self._last_tick)
        self._last_tick = now
        self.label_fps_info["text"] = f"{self.fps:.2f}"

    def create_face_folder(self):
        #  Create the folders for saving faces
        self.existing_faces_cnt += 1
        self.current_face_dir = os.path.join(
            self.path_photos_from_camera, f"person_{self.existing_faces_cnt}_{self.input_name_char}")
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = f"\"{self.current_face_dir}\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)

        self.ss_cnt = 0  # Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created

    def save_current_face(self):
        if not self.face_folder_created_flag:
            self.log_all["text"] = "Please run step 2 first!"
            return
        # Exactly one face: with several, the ROI would be whichever face was
        # detected last, which may not be the student being registered.
        if self.current_frame_faces_cnt > 1:
            self.log_all["text"] = "Multiple faces in frame - only the student being registered should be visible!"
            return
        if self.current_frame_faces_cnt == 0 or self.current_frame_clean is None:
            self.log_all["text"] = "No face detected in current frame!"
            return

        t_start = time.perf_counter()
        # Safe boundary slicing
        h_img, w_img = self.current_frame_clean.shape[:2]
        y1 = max(0, self.face_ROI_height_start - self.hh)
        y2 = min(h_img, self.face_ROI_height_start + self.face_ROI_height + self.hh)
        x1 = max(0, self.face_ROI_width_start - self.ww)
        x2 = min(w_img, self.face_ROI_width_start + self.face_ROI_width + self.ww)
        face_ROI_image = self.current_frame_clean[y1:y2, x1:x2]

        save_path = os.path.join(self.current_face_dir, f"img_face_{self.ss_cnt + 1}.jpg")
        if not imwrite_rgb(save_path, face_ROI_image):
            self.log_all["text"] = f"Could not save \"{save_path}\"!"
            return
        self.ss_cnt += 1
        logging.info("Save into: %s", save_path)

        note = ""
        if self.reference_desc is None:
            self.reference_desc = self.current_desc
        elif self.live_distance is not None and self.live_distance >= MATCH_THRESHOLD:
            note = (f"\nWarning: this face looks different from the first photo "
                    f"(distance {self.live_distance:.2f}). Retake if it is not the same student.")
        self.log_all["text"] = f"\"{save_path}\" saved! ({self.ss_cnt}/{PHOTOS_PER_STUDENT}){note}"

        # Capture latency for the admin dashboard graph: face detection +
        # descriptor time of this frame plus the time to crop and save it.
        capture_ms = round(self.process_ms + (time.perf_counter() - t_start) * 1000, 1)
        self.capture_latencies.append(capture_ms)
        try:
            reg_data = {
                "registered": True,
                "student_folder": self.input_name_char or "New Student",
                "photos_count": len(self.capture_latencies),
                "latencies": self.capture_latencies,
                "avg_latency": round(sum(self.capture_latencies) / len(self.capture_latencies), 1),
                "last_capture_ms": capture_ms
            }
            with open(common.LATENCY_JSON, "w", encoding="utf-8") as f:
                json.dump(reg_data, f, indent=2)
        except OSError as ex:
            logging.warning("Could not save registration_latency.json: %s", ex)

        if self.ss_cnt >= PHOTOS_PER_STUDENT and not self.extraction_launched:
            # Fire once; later photos are picked up by the next rebuild.
            self.extraction_launched = True
            self.log_all["text"] = f"{PHOTOS_PER_STUDENT} Photos saved! Feature database updating..."
            subprocess.Popen([sys.executable, 'features_extraction_to_csv.py'], cwd=common.BASE_DIR)

    def get_frame(self):
        # Always return a 2-tuple: the caller unpacks this directly, so a bare
        # None here raises TypeError the moment the camera hiccups.
        try:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None
            frame = cv2.resize(frame, (640, 480))
            return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            logging.warning("Error: No video input!!! (%s)", e)
            return False, None

    def process_frame(self, frame):
        """Detect the face(s) in an RGB frame and draw the HUD onto self.current_frame."""
        self.current_frame = frame
        self.current_frame_clean = frame.copy()
        t0 = time.perf_counter()
        faces = detect_faces(self.detector, frame)
        self.current_frame_faces_cnt = len(faces)
        self.current_desc = None
        self.live_distance = None
        if len(faces) == 1:
            self.current_desc = face_descriptor(self.predictor, self.face_reco_model, frame, faces[0])
            if self.reference_desc is not None:
                self.live_distance = float(np.linalg.norm(self.current_desc - self.reference_desc))
        self.process_ms = (time.perf_counter() - t0) * 1000

        self.update_fps()
        self.label_face_cnt["text"] = str(len(faces))
        hud = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Processing: {self.process_ms:.1f} ms  ({self.fps:.1f} FPS)",
                    (15, 25), hud, 0.6, (0, 255, 0), 2)

        if len(faces) == 1:
            d = faces[0]
            self.face_ROI_width_start = d.left()
            self.face_ROI_height_start = d.top()
            self.face_ROI_height = d.bottom() - d.top()
            self.face_ROI_width = d.right() - d.left()
            self.hh = int(self.face_ROI_height / 2)
            self.ww = int(self.face_ROI_width / 2)
            if self.reference_desc is None:
                status, color = "Face OK - save the first photo", (0, 255, 0)
            elif self.live_distance < MATCH_THRESHOLD:
                status, color = f"Same person as 1st photo (dist {self.live_distance:.3f})", (0, 255, 0)
            else:
                status, color = f"Differs from 1st photo (dist {self.live_distance:.3f})", (255, 165, 0)
            cv2.rectangle(frame,
                          (max(0, d.left() - self.ww), max(0, d.top() - self.hh)),
                          (min(640, d.right() + self.ww), min(480, d.bottom() + self.hh)),
                          color, 2)
            cv2.putText(frame, status, (15, 50), hud, 0.6, color, 2)
            self.label_warning["text"] = ""
        elif len(faces) > 1:
            for d in faces:
                cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)
            cv2.putText(frame, "Multiple faces - only one person please", (15, 50), hud, 0.6, (255, 0, 0), 2)
            self.label_warning["text"] = "Only one person should be in front of the camera."
        else:
            cv2.putText(frame, "Searching for face...", (15, 50), hud, 0.6, (255, 0, 0), 2)
            self.label_warning["text"] = ""

    #  Main process of face detection and saving
    def process(self):
        ret, frame = self.get_frame()
        if not ret or frame is None:
            self.label_warning["text"] = "No camera frame - check the webcam."
            self.win.after(200, self.process)
            return
        self.process_frame(frame)

        # Convert PIL.Image.Image to PIL.Image.PhotoImage
        img_PhotoImage = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
        self.label.img_tk = img_PhotoImage
        self.label.configure(image=img_PhotoImage)

        # Refresh frame
        self.win.after(20, self.process)

    def open_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        return cap

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.win.destroy()

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.cap = self.open_camera()
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.process()
        self.win.mainloop()


def main():
    logging.basicConfig(level=logging.INFO)
    init_db()
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
