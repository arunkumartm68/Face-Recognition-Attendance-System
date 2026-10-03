# Extract features from the saved face photos into "data/features_all.csv":
# one averaged 128D descriptor per registered student.

import logging
import os

import numpy as np

import common
from face_utils import (detect_faces, face_descriptor, imread_rgb, largest_face,
                        load_models, photo_fingerprint, write_features)


def return_features_list_personX(path_face_personX, models):
    """128D descriptors of every usable photo in one student's folder."""
    detector, predictor, model = models
    features_list_personX = []
    photos_list = sorted(os.listdir(path_face_personX))
    if not photos_list:
        logging.warning("Warning: No images in %s/", path_face_personX)
    for photo in photos_list:
        if not photo.lower().endswith(common.IMAGE_EXTENSIONS):
            continue
        img_path = os.path.join(path_face_personX, photo)
        logging.info("Reading image: %s", img_path)
        rgb = imread_rgb(img_path)
        if rgb is None:
            logging.warning("Could not read %s", img_path)
            continue
        faces = detect_faces(detector, rgb)
        if not faces:
            logging.warning("No face detected in %s", img_path)
            continue
        features_list_personX.append(face_descriptor(predictor, model, rgb, largest_face(faces)))
    return features_list_personX


def main():
    logging.basicConfig(level=logging.INFO)
    models = load_models()
    # "Process Faces" can be clicked before any face was ever registered
    os.makedirs(common.FACES_DIR, exist_ok=True)
    # Taken before reading: photos saved while this runs will not match it,
    # so the next check rebuilds again and picks them up.
    fingerprint = photo_fingerprint()

    rows = []
    for person in sorted(os.listdir(common.FACES_DIR)):
        folder_path = os.path.join(common.FACES_DIR, person)
        if not os.path.isdir(folder_path):
            continue
        features_list = return_features_list_personX(folder_path, models)
        if not features_list:
            logging.warning("Skipping %s: No valid images found.", folder_path)
            continue
        label = common.folder_label(person)
        rows.append([label] + list(np.array(features_list, dtype=float).mean(axis=0)))
        logging.info("Saved profile vector for: %s (%d photos)", label, len(features_list))

    write_features(rows, fingerprint)
    logging.info("Successfully updated %d face profile(s) into: %s", len(rows), common.FEATURES_CSV)


if __name__ == '__main__':
    main()
