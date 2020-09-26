from src.common.preprocess_image import get_face_attribute, align_image
import cv2
import tensorflow as tf
from mtcnn import MTCNN


def run_preprocess():
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
