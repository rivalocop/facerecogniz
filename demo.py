# import the necessary packages
import pickle

import cv2
import imutils
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

import settings
from src.align.face_detector import detect_face, align_face
from src.embedding.embedding import Embedding
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
    # recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
    # le = pickle.loads(open('output/le.pickle', "rb").read())
    recognizer = pickle.loads(open('output/tf_recognizer.pickle', "rb").read())
    le = pickle.loads(open('output/tf_le.pickle', "rb").read())
    detector = MTCNN()
    tf_model = Embedding(model_fp=settings.FACE_DESCRIBER_MODEL_FP,
                         input_tensor_names=settings.FACE_DESCRIBER_INPUT_TENSOR_NAMES,
                         output_tensor_names=settings.FACE_DESCRIBER_OUTPUT_TENSOR_NAMES,
                         device=settings.FACE_DESCRIBER_DEVICE)
    image = cv2.imread(
        'demo1.jpg')
    image = imutils.resize(image, width=600)
    face = detect_face(detector, image)
    if face is not None:
        aligned = align_face(image, desiredLeftEye=(0.4, 0.4),
                             left_eye_coordinates=face.keypoints.left_eye,
                             right_eye_coordinates=face.keypoints.right_eye)
        aligned_face = detect_face(detector, aligned)
        if aligned_face is not None:
            result = aligned[
             aligned_face.box[1]:aligned_face.box[1] + aligned_face.box[3],
             aligned_face.box[0]:aligned_face.box[0] + aligned_face.box[2]
            ]
            output = cv2.resize(result, (112, 112))
            dropout_rate = 0.5
            input_data = [np.expand_dims(output, axis=0), dropout_rate]
            face_embedding = tf_model.inference(data=input_data)

            preds = recognizer.predict_proba(face_embedding[0])[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text = "{}: {:.2f}%".format(name, proba * 100)
            y = face.box[1] - 10 if face.box[1] - 10 > 10 else face.box[1] + 10
            cv2.rectangle(image, (face.box[0], face.box[1]), (face.box[0] + face.box[2], face.box[1] + face.box[3]),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (face.box[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imwrite('test_demo1.png', image)
