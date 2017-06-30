import numpy as np
import logging
import mxnet as mx
import pandas as pd

logging.basicConfig(level=logging.INFO)

BATCH = 1024
DEVS = mx.gpu()


def net_symbol(num_class=5089):
    net = mx.sym.Variable('data')
    fc1 = mx.symbol.FullyConnected(data=net, num_hidden=num_class, name='fc1_1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')


def get_data():
    trn_x = np.load('tmp/resneXt101_trn_0_pool.npy')
    val_x = np.load('tmp/resneXt101_val_0_pool.npy')

    trn_y = np.load('tmp/resneXt101_trn_0_lbs.npy')
    val_y = np.load('tmp/resneXt101_val_0_lbs.npy')
    print(trn_x.shape, trn_y.shape)
    print(trn_y[:10], val_y[:10])

    trn = mx.io.NDArrayIter(data=trn_x, label=trn_y, batch_size=BATCH, shuffle=True)
    del trn_x
    val = mx.io.NDArrayIter(data=val_x, label=val_y, batch_size=BATCH)
    del val_x

    return trn, val


def train_mlp():
    net = net_symbol()
    model = mx.mod.Module(symbol=net)

    trn, val = get_data()

    save_model_prefix = 'model/mlp-'
    checkpoint = [mx.callback.do_checkpoint(save_model_prefix)]
    eval_metrics = ['accuracy', 'ce']
    for top_k in [5, 10, 20]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=top_k))

    batch_end_callback = [mx.callback.Speedometer(BATCH, 30)]

    model.fit(trn, val,
              eval_metric=eval_metrics, optimizer_params={'learning_rate': 0.3, 'momentum': 0.9},
              num_epoch=100,
              batch_end_callback=batch_end_callback,
              epoch_end_callback=checkpoint
              )


if __name__ == '__main__':
    train_mlp()
