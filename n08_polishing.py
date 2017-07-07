import mxnet as mx
import numpy as np
import importlib
import os, sys
import logging

# logging.basicConfig(level=logging.DEBUG)

log_file = "logs/resnext-101-log"
log_dir = "./"
log_file_full_name = os.path.join(log_dir, log_file)
head = '%(asctime)-15s Node[' + str(mx.kvstore.create("local").rank) + '] %(message)s'

logger = logging.getLogger()
handler = logging.FileHandler(log_file_full_name)
formatter = logging.Formatter(head)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

TRAIN_BIN = 'rec/train'  # 'openimages_train.rec'/media/devbox/storage3/
VAL_BIN = "rec/val.rec"  # 'openimages_val.rec'
DATA_SHAPE = (3, 224, 224)

# ====================================================

NUM_GPU = 4
BATCH_SIZE = 112
LR = 0.0001

PROCESS = 'retrain'
START_EPOCH = 30
NET_TYPE = 'resnext'
DEPTH = 101

# ====================================================

kv = mx.kvstore.create("local")
devs = []
for i in range(NUM_GPU):
    devs.append(mx.gpu(i))


def finetune_symbol(symbol, model, **kwargs):
    initializer = mx.initializer.Load(param=model.arg_params, default_init=mx.init.Uniform(0.001))
    new_model = mx.model.FeedForward(symbol=symbol, initializer=initializer, **kwargs)
    return new_model


def get_train_val(train_bin=TRAIN_BIN,
                  val_bin=VAL_BIN,
                  batch_size=BATCH_SIZE,
                  data_shape=DATA_SHAPE,
                  aug=True, mean=False):
    train_path = train_bin
    val_path = val_bin

    train = mx.io.ImageRecordIter(
        path_imgrec=train_path,
        mean_r=128.0 if mean else 0.0,
        mean_g=128.0 if mean else 0.0,
        mean_b=128.0 if mean else 0.0,

        shuffle=True,
        rand_crop=True,
        rand_mirror=True,

        # max_rotate_angle=10 if aug else 0,
        # max_aspect_ratio=0.25 if aug else 0,
        # max_shear_ratio=0.1 if aug else 0,

        # # max_random_scale=1.0 if aug else 1.0,
        # # min_random_scale=0.85 if aug else 1.0,

        # random_h=36 if aug else 0,
        # random_s=40 if aug else 0,
        # random_l=40 if aug else 0,

        prefetch_buffer=2,
        data_shape=data_shape,
        batch_size=batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        preprocess_threads=8)

    val = mx.io.ImageRecordIter(
        path_imgrec=val_path,
        mean_r=128.0 if mean else 0.0,
        mean_g=128.0 if mean else 0.0,
        mean_b=128.0 if mean else 0.0,

        rand_crop=False,
        rand_mirror=False,

        prefetch_buffer=1,
        data_shape=data_shape,
        batch_size=batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        preprocess_threads=8)

    return train, val


def train(epoch=7, batch_size=BATCH_SIZE):
    model_path = 'model/resnext101'
    print('model will save to %s' % model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    save_model_prefix = model_path + '/-0'

    model_args = {}
    model_args['epoch_size'] = 500  # iteration

    # symbol = memonger.search_plan(symbol)
    wd = 0.0001
    momentum = 0.9

    # load_model_prefix = "models_weights/%s/%s-%s/-0" % ('cloth_4k', net_type, depth)
    load_model_prefix = 'model/resnext101/-0'
    pretrained_model = mx.model.FeedForward.load(load_model_prefix, 30, ctx=mx.cpu())
    symbol = pretrained_model.symbol

    model = finetune_symbol(symbol, pretrained_model, optimizer='nag', begin_epoch=epoch,
                            learning_rate=LR, momentum=momentum, wd=wd, num_epoch=500, ctx=devs, **model_args)

    eval_metrics = ['accuracy', 'ce']
    for top_k in [5]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=top_k))

    batch_end_callback = [mx.callback.Speedometer(batch_size, 500)]
    checkpoint = mx.callback.do_checkpoint(save_model_prefix)

    train, val = get_train_val()
    logging.info('LR %s, WD %s, momentum %s' %
                 (str(LR), str(wd), str(momentum)))
    logging.info('Batch size %s' % str(BATCH_SIZE))

    model.fit(
        X=train,
        eval_data=val,
        eval_metric=eval_metrics,
        kvstore='local_allreduce_device',
        batch_end_callback=batch_end_callback,
        epoch_end_callback=checkpoint)


if __name__ == '__main__':
    print('!!!', LR, START_EPOCH, PROCESS)
    train(epoch=START_EPOCH)

    # train(process='finetune', net_type='resnet', depth=50, data_part='train', epoch = 7)
