import cv2


class OpenFaceModel:

    def __init__(self, model_path):
        self.detector = self.__load_model(model_path)

    def __load_model(self, model_path):
        return cv2.dnn.readNetFromTorch(model_path)

    def get_feature(self, face_cropped):
        blob = cv2.dnn.blobFromImage(
            face_cropped, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.detector.setInput(blob)
        vec = self.detector.forward()
        return vec
