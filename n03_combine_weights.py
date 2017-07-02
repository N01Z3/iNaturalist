import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt


def main():
    pretrained_model = mx.model.FeedForward.load(
        '/media/aakuzin/DATA/dataset/ModelZoo/imagenet11k_places365/resnet-152',
        0, ctx=mx.cpu())
    pretrain_fc = mx.model.FeedForward.load("model/mlp-", 10, ctx=mx.cpu())

    internals = pretrained_model.symbol.get_internals()  # .list_outputs()
    print(internals.list_outputs()[-15:])
    fea_symbol = internals["flatten0_output"]

    fc1 = mx.symbol.FullyConnected(data=fea_symbol, num_hidden=5089, name='fc1_1')
    symbol = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

    z = pretrained_model.arg_params.copy()
    z.update(pretrain_fc.arg_params)

    mx.model.save_checkpoint('model/resnet152_11p', 1, symbol, z, pretrained_model.aux_params)


def check():
    symbol, arg_params, aux_params = mx.model.load_checkpoint('model/resnet152_11p', 1)

    fc = arg_params['fc1_weight'].asnumpy()

    plt.plot(fc)
    plt.show()

    # mod = mx.module.Module(symbol)
    # mod.bind(data_shapes=[('data', (1, 3, 224, 224))], for_training=False)
    # mod.init_params(arg_params=arg_params, aux_params=aux_params)
    #
    # image = 0.5*np.ones((1,224,224,3))
    #
    # x = mx.io.NDArrayIter(np.transpose(image, (0, 3, 1, 2)))
    # out = mod.predict(x).asnumpy()
    #
    # print('mxnet:', out.argmax(), out[0, out.argmax()], out.shape)


if __name__ == '__main__':
    main()
    check()
