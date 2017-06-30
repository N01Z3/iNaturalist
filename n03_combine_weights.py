import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import logging

# def finetune_symbol_2model(symbol, model_bd, model_fc, **kwargs):
#     z = model_bd.arg_params.copy()
#     z.update(model_fc.arg_params)
#
#     initializer = mx.initializer.Load(param=z, default_init=mx.init.Uniform(0.001))
#     new_model = mx.model.FeedForward(symbol=symbol, initializer=initializer, **kwargs)
#     return new_model

def main():
    pretrained_model = mx.model.FeedForward.load('/media/aakuzin/DATA/dataset/ModelZoo/imagenet1k/resnext/resnext-101',
                                                 0, ctx=mx.cpu())
    pretrain_fc = mx.model.FeedForward.load("model/mlp-", 6, ctx=mx.cpu())

    internals = pretrained_model.symbol.get_internals()#.list_outputs()
    print(internals.list_outputs()[-15:])
    fea_symbol = internals["flatten0_output"]

    fc1 = mx.symbol.FullyConnected(data=fea_symbol, num_hidden=5089, name='fc1_1')
    symbol = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

    # model = finetune_symbol_2model(symbol, pretrained_model, pretrain_fc)

    z = pretrained_model.arg_params.copy()
    z.update(pretrain_fc.arg_params)

    # model.save('model/resnext101', 0)
    mx.model.save_checkpoint('model/resnext101', 1, symbol, z, pretrained_model.aux_params)


def check():
    symbol, arg_params, aux_params = mx.model.load_checkpoint('model/resnext101', 1)

    fc= arg_params['fc1_weight'].asnumpy()

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
    # main()
    check()