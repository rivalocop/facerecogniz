import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from mtcnn import MTCNN

import settings
from src.align.face_detector import detect_face, align_face
from src.embedding.embedding import Embedding

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


def run_extract_embedding():
    detector = MTCNN()
    # fm = FaceModel(settings.DEEPSIGHTFACE_DIR)
    tf_model = Embedding(model_fp=settings.FACE_DESCRIBER_MODEL_FP,
                         input_tensor_names=settings.FACE_DESCRIBER_INPUT_TENSOR_NAMES,
                         output_tensor_names=settings.FACE_DESCRIBER_OUTPUT_TENSOR_NAMES,
                         device=settings.FACE_DESCRIBER_DEVICE)
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
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
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
                prediction = tf_model.inference(data=input_data)
                knownNames.append(name)
                knownEmbeddings.append(prediction[0].flatten())
                total += 1
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("output/tf_embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == '__main__':
    run_extract_embedding()
