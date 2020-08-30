import numpy as np
import mxnet as mx
from sklearn.preprocessing import normalize


class FaceModel:
    def __init__(self, model_path):
        ctx = mx.gpu(0)
        image_size = (112, 112)
        self.model = self.__get_model(ctx, image_size, model_path, 'fc1')

        self.threshold = 1.24
        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        # self.det_factor = 0.9
        self.image_size = image_size

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = normalize(embedding).flatten()
        return embedding

    def __get_model(self, ctx, image_size, model_str, layer):
        _vec = model_str.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer + '_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[(
        # 'softmax_label', (args.batch_size,))])
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        return model
