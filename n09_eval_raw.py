import mxnet as mx
import numpy as np
import os, sys
import logging
import argparse
from time import time
import pandas as pd
from sklearn.metrics import accuracy_score
import cv2
import glob
from multiprocessing import Pool


def preprocess_image(path):
    # load image
    img = cv2.imread(path)
    if img is None:
        print('ololo')
        return np.zeros((1, 3, 320, 320))
    img = img[:, :, [2, 1, 0]]

    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    img = img[yy: yy + short_egde, xx: xx + short_egde]

    img = cv2.resize(img, (320, 320))

    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2).astype(np.float32)

    # img -= 117
    return img.reshape([1, 3, 320, 320])


def get_mod(netwk, epoch):
    sym, arg_params, aux_params = mx.model.load_checkpoint(netwk, epoch)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu())
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 320, 320))])
    mod.set_params(arg_params, aux_params)
    return mod


def main2():

    mod1 = get_mod('model/resnext101t/-0', 49)
    mod2 = get_mod('model/resnext101/-0', 42)
    mod3 = get_mod('model/resnet152k/-0', 45)

    df = pd.read_csv('data/test.lst', sep='\t', header=None, names=['0', '1', 'fns'])
    fns = [os.path.join('../test2017', fn) for fn in df['fns'].tolist()]

    pds = []
    cnt = 0
    t0 = time()
    for i in range(0, len(fns), 5000):
        c_fns = fns[i:i + 5000]

        with Pool() as pool:
            samples = pool.map(preprocess_image, c_fns)

        samples = np.vstack(samples)
        print(samples.shape)
        samples = mx.io.NDArrayIter(samples)

        feats1 = mod1.predict(samples).asnumpy()
        samples.reset()

        feats2 = mod2.predict(samples).asnumpy()
        samples.reset()

        feats3 = mod3.predict(samples).asnumpy()

        feats = np.array([feats1, feats2, feats3])

        print(feats1.shape, time() - t0)
        np.save(os.path.join('tmp', '%s_%d_%d' % ('320', 66, i)), feats)
        t0 = time()


if __name__ == '__main__':
    main2()
