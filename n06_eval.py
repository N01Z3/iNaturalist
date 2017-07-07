import mxnet as mx
import numpy as np
import os, sys
import logging
import argparse
from time import time
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import gmean

print(mx.__version__)
os.environ['MXNET_BACKWARD_DO_MIRROR'] = '0'
MODELZOO = "./model"


def add_fit_args(args):
    args.add_argument('--netwk', default=MODELZOO + '/resnext101/-0', type=str, help='net symbol path')
    args.add_argument('--begin', default=30, type=int, help='start epoch')
    args.add_argument('--input', default='3,224,224', type=str, help='shape of net input')
    args.add_argument('--flatn', default="flatten0_output", type=str, help='name of global pooling layer')
    args.add_argument('--means', default="0,0,0", type=str, help='substract from image channels')
    args.add_argument('--scale', default=1.0, type=float, help='scale all image channels')

    args.add_argument('--dataf', default='../val.rec', type=str,
                      help='data file for inference')
    args.add_argument('--chunk', default=200000, type=int, help='dump size')

    args.add_argument('--ngpus', default=0, type=int, help='gpu number')
    args.add_argument('--batch', default=32, type=int, help='the batch size')
    args.add_argument('--kvstr', default='device', type=str, help='key-value store type')

    args.add_argument('--pdump', default='./tmp', type=str, help='path to dumps predictions')
    args.add_argument('--phead', default='esnext-101_val', type=str, help='name of dumps files')
    return args


def get_data(args, kv):
    val_rec = args.dataf
    r, g, b = [int(a) for a in args.means.split(',')]
    data_shape = tuple([int(a) for a in args.input.split(',')])

    val = mx.io.ImageRecordIter(
        path_imgrec=val_rec,
        mean_r=r,
        mean_g=g,
        mean_b=b,
        scale=args.scale,

        max_rotate_angle=5,
        max_aspect_ratio=0.1,
        max_shear_ratio=0.05,

        random_h=10,
        random_s=10,
        random_l=10,

        rand_crop=True,
        rand_mirror=True,

        prefetch_buffer=2,
        data_shape=data_shape,
        batch_size=args.batch,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        preprocess_threads=4)

    return val


def main(args):
    kv = mx.kvstore.create(args.kvstr)

    data = get_data(args, kv)

    dev = mx.gpu(args.ngpus)

    sym, arg_params, aux_params = mx.model.load_checkpoint(args.netwk, args.begin)
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training=False, data_shapes=data.provide_data)
    mod.set_params(arg_params, aux_params)

    pds, lbs = [], []
    cnt = 0
    t0 = time()
    for preds, i_batch, batch in mod.iter_predict(data):
        pds.append(preds[0].asnumpy())
        lbs.append(batch.label[0].asnumpy().astype('int8'))
        if len(pds * args.batch) > args.chunk:
            pds = np.vstack(pds)
            # lbs = np.concatenate(lbs)
            print(time() - t0, pds.shape, pds[0])
            np.save(os.path.join(args.pdump, '%s_%d_%d' % (args.phead, args.begin, cnt)), pds)
            pds, lbs, t0 = [], [], time()
            cnt += 1

    if len(pds) > 0:
        pds = np.vstack(pds)
        np.save(os.path.join(args.pdump, '%s_%d_%d' % (args.phead, args.begin, cnt)), pds)


def check_acc():
    y_true = \
        pd.read_csv('data/val.lst',
                    sep='\t', header=None)[1].as_matrix()

    y_pred = []
    for fix in ['50_257', '101_342', '101_497']:
        y_pred.append(np.load('tmp/resnext%s_0.npy' % fix))

    y_pred = gmean(np.array(y_pred), axis=0)

    print(y_true.shape, y_pred.shape)
    print(accuracy_score(y_true, np.argmax(y_pred, axis=1)))


if __name__ == '__main__':
    if not os.path.exists('tmp'): os.mkdir('tmp')
    parser = argparse.ArgumentParser()
    parser = add_fit_args(parser)
    ags = parser.parse_args()
    main(ags)
    # check_acc()
