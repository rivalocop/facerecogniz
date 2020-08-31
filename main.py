from src.facenet.facenet_model import FacenetModel
import cv2
import tensorflow.compat.v1 as tf
from mtcnn import MTCNN
import numpy as np
from skimage import transform as trans

from src.align.face_detector import detect_face, align_face


tf.disable_v2_behavior()
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


def run_preprocess():
    facenet_md = FacenetModel()
    # detector = MTCNN()
    image_path = 'example1.jpg'
    # image = cv2.imread(image_path)
    facenet_md.align(image_path)
    # face = detect_face(detector, image)
    # aligned = align_face(image, desiredLeftEye=(0.35, 0.35),
    #                      left_eye_coordinates=face.keypoints.left_eye,
    #                      right_eye_coordinates=face.keypoints.right_eye)
    # output = cv2.resize(aligned, (112, 112))
    # cv2.imwrite('temp.png', output)
    # print(face.confidence)


if __name__ == '__main__':
    run_preprocess()
