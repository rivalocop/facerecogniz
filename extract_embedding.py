import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from mtcnn import MTCNN

import settings
from src.common.preprocess_image import get_face_attribute, align_image
from src.embedding.face_model import FaceModel

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1700)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def run_extract_embedding():
    detector = MTCNN()
    fm = FaceModel(settings.DEEPSIGHTFACE_DIR)
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images('dataset/train'))
    knownEmbeddings = []
    knownNames = []
    total = 0
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        if image is None:
            continue
        if len(image) == 0:
            continue
        faces = detector.detect_faces(image)
        if len(faces) > 0:
            face = faces[0]
            bbox, points = get_face_attribute(face)
            aligned_image = align_image(image, bbox, points)
            output = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
            output = np.transpose(output, (2, 0, 1))
            f1 = fm.get_feature(output)
            knownNames.append(name)
            knownEmbeddings.append(f1)
            total += 1
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("output/mxnet_embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == '__main__':
    run_extract_embedding()
