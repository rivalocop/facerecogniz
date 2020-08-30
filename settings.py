import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEEPSIGHTFACE_DIR = BASE_DIR + '/models/model-r100-ii/model,0'
FACE_DESCRIBER_INPUT_TENSOR_NAMES = ['img_inputs:0', 'dropout_rate:0']
FACE_DESCRIBER_OUTPUT_TENSOR_NAMES = ['resnet_v1_50/E_BN2/Identity:0']
FACE_DESCRIBER_DEVICE = '/cpu:0'
FACE_DESCRIBER_MODEL_FP = BASE_DIR + '/models/insightface.pb'
FACE_DESCRIBER_TENSOR_SHAPE = (112, 112)
FACE_DESCRIBER_DROP_OUT_RATE = 0.1

FACE_SIMILARITY_THRESHOLD = 800
