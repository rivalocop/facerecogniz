from src.common.preprocess_image import get_face_attribute, align_image
import cv2
import tensorflow as tf
from mtcnn import MTCNN


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
    # facenet_md = FacenetModel()
    detector = MTCNN()
    image_path = 'example1.jpg'
    image = cv2.imread(image_path)
    faces = detector.detect_faces(image)
    if len(faces) > 0:
        face = faces[0]
        bbox, points = get_face_attribute(face)
        aligned_image = align_image(image, bbox, points)
        cv2.imwrite('temp.png', aligned_image)


if __name__ == '__main__':
    run_preprocess()
