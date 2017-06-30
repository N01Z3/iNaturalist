import numpy as np
import logging
import mxnet as mx
import os

logging.basicConfig(level=logging.INFO)

BATCH = 256
DEVS = mx.gpu()


def net_symbol(num_class=5089):
    net = mx.sym.Variable('data')
    fc1 = mx.symbol.FullyConnected(data=net, num_hidden=num_class, name='fc1_1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')


def get_data():
    trn_x = np.load('tmp/resneXt101_val_0_pool.npy')
    val_x = np.load('tmp/resneXt101_val_0_pool.npy')

    trn_y = np.load('tmp/resneXt101_val_0_lbs.npy')
    val_y = np.load('tmp/resneXt101_val_0_lbs.npy')
    print(trn_x.shape, trn_y.shape, val_x.shape, val_y.shape,)
    print(trn_y[:10], val_y[:10])

    trn = mx.io.NDArrayIter(data=trn_x, label=trn_y, batch_size=BATCH, shuffle=True)
    del trn_x
    val = mx.io.NDArrayIter(data=val_x, label=val_y, batch_size=BATCH)
    del val_x

    return trn, val


def train_mlp():
    net = net_symbol()
    model = mx.mod.Module(symbol=net, context=DEVS)

    trn, val = get_data()

    save_model_prefix = 'model/mlp-'

    if os.path.isfile(save_model_prefix + '-symbol.json'):
        sym, arg_params, aux_params = mx.model.load_checkpoint(save_model_prefix, 2)

    checkpoint = [mx.callback.do_checkpoint(save_model_prefix)]
    eval_metrics = ['accuracy', 'ce']
    for top_k in [5, 10, 20]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=top_k))

    epoch_size = 115837
    steps = [epoch_size * (x - 0) for x in [2, 10, 15, 20] if x - 0 > 0]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.1)
    batch_end_callback = [mx.callback.Speedometer(BATCH, 30)]

    model.fit(trn, val,
              eval_metric=eval_metrics,
              optimizer_params={'learning_rate': 0.03, 'momentum': 0.9, 'lr_scheduler': lr_scheduler},
              num_epoch=100,
              batch_end_callback=batch_end_callback, arg_params=arg_params, aux_params=aux_params,
              epoch_end_callback=checkpoint
              )


if __name__ == '__main__':
    train_mlp()
