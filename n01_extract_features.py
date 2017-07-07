import mxnet as mx
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import cv2
from time import time
from multiprocessing import Pool
import pandas as pd
from shutil import copyfile
import pickle
from scipy.stats import gmean

BATCH_SIZE = 20
DATA_SHAPE = (3, 224, 224)
TENR_SHAPE = (BATCH_SIZE, DATA_SHAPE[0], DATA_SHAPE[1], DATA_SHAPE[2])
kv = mx.kvstore.create("local")


def get_model(prefix='model_zoo/avitonet_fasion/0/-1', epoch=24):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    # sym = mx.sym.Group([sym.get_internals()['{}_output'.format(x)]
    #                     for x in ['flatten0', 'softmax']])
    sym = sym.get_internals()['flatten0_output']

    mod = mx.mod.Module(symbol=sym, context=mx.gpu())
    mod.bind(data_shapes=[('data', TENR_SHAPE)], for_training=False)
    mod.set_params(arg_params, aux_params)
    return mod


def get_data(rec_path):
    val = mx.io.ImageRecordIter(
        path_imgrec=rec_path,
        mean_r=0.0,
        mean_g=0.0,
        mean_b=0.0,

        rand_crop=False,
        rand_mirror=False,

        prefetch_buffer=2,
        data_shape=DATA_SHAPE,
        batch_size=BATCH_SIZE,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        preprocess_threads=7)
    return val


def predict_rec(model_json, epoch, rec_path, out_file):
    model = get_model(model_json, epoch)
    data = get_data(rec_path)

    lbs, pll = [], []
    cnt = 0
    t0 = time()
    for preds, i_batch, batch in model.iter_predict(data):
        pool = preds[0].asnumpy()

        pll.append(pool)
        lbs.append(batch.label[0].asnumpy().astype('int8'))
        # if len(pll) > 10000:
    pll = np.vstack(pll)
    lbs = np.concatenate(lbs)  # .flatten()
    print(time() - t0, pll.shape, lbs.shape, pll[0])
    np.save('tmp/%s_%d_lbs' % (out_file, cnt), lbs)
    np.save('tmp/%s_%d_pool' % (out_file, cnt), pll)
    # np.save('preds/trn2_label_%d' % cnt, lbs)
    lbs, pll, t0 = [], [], time()
    cnt += 1


if __name__ == '__main__':
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet1k/resnext/resnext-50', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/val_s.rec', 'resneXt50_val')
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet1k/resnext/resnext-50', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/train_pt0.rec', 'resneXt50_trn')
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet1k/resnext/resnext-101', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/val_s.rec', 'resneXt101_val')
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet1k/resnext/resnext-101', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/train_pt0.rec', 'resneXt101_trn')
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet11k_places365/resnet-152', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/val_s.rec', 'resnet152_1k_val')
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet11k_places365/resnet-152', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/train/train_pt0.rec', 'resnet152_1k_trn')

    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet11k/resnet-152', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/val_s.rec', 'resnet152_11k_val')
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet11k/resnet-152', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/train/train_pt0.rec', 'resnet152_11k_trn')
    #
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet1k/resnext/resnext-101-64x4d', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/val_s.rec', 'resneXt101_64_val')
    # predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/imagenet1k/resnext/resnext-101-64x4d', 0,
    #             '/media/aakuzin/DATA/download/iNaturalist/train_pt0.rec', 'resneXt101_64_trn')
    predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/Avitonet4k/resnext-101/-0', 497,
                '/media/aakuzin/DATA/download/iNaturalist/train/train_pt0.rec', 'resneXt101a_trn')
    predict_rec('/media/aakuzin/DATA/dataset/ModelZoo/Avitonet4k/resnext-101/-0', 497,
                '/media/aakuzin/DATA/download/iNaturalist/val_s.rec', 'resneXt101a_val')