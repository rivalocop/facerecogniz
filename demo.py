# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import tensorflow as tf
from mtcnn import MTCNN

import settings
from src.common.preprocess_image import get_face_attribute, align_image
from src.embedding.face_model import FaceModel

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(
            logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if __name__ == '__main__':
    recognizer = pickle.loads(
        open('output/mxnet_recognizer.pickle', "rb").read())
    le = pickle.loads(open('output/mxnet_le.pickle', "rb").read())
    detector = MTCNN()
    image = cv2.imread(
        'demo1.jpg')
    image = imutils.resize(image, width=600)
    faces = detector.detect_faces(image)
    if len(faces) > 0:
        face = faces[0]
        bbox, points = get_face_attribute(face)
        aligned_image = align_image(image, bbox, points)
        output = cv2.resize(aligned_image, (112, 112))
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = np.transpose(output, (2, 0, 1))

        fm = FaceModel(settings.DEEPSIGHTFACE_DIR)
        f1 = fm.get_feature(output)
        f1 = np.expand_dims(f1, axis=0)

        preds = recognizer.predict_proba(f1)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        text = "{}: {:.2f}%".format(name, proba * 100)
        y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (bbox[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imwrite('test_demo1.png', image)
