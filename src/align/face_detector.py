from typing import List, Tuple

import numpy as np
import cv2
from mtcnn import MTCNN
from pydantic import BaseModel


def detect_face(detector: MTCNN, image):
    result = detector.detect_faces(image)
    if len(result) > 0:
        face = FaceAttribute(**result[0])
        return face
    else:
        return None


def align_face(image, left_eye_coordinates,
               right_eye_coordinates,
               desiredLeftEye=(0.35, 0.35),
               desiredFaceWidth=256,
               desiredFaceHeight=None):
    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    leftEyeCenter = np.asarray(left_eye_coordinates)
    rightEyeCenter = np.asarray(right_eye_coordinates)

    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    angle = np.degrees(np.arctan2(dY, dX))

    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_CUBIC)
    return output


class Keypoint(BaseModel):
    nose: Tuple[int, int]
    mouth_right: Tuple[int, int]
    right_eye: Tuple[int, int]
    left_eye: Tuple[int, int]
    mouth_left: Tuple[int, int]


class FaceAttribute(BaseModel):
    box: List[int]
    keypoints: Keypoint
    confidence: float
