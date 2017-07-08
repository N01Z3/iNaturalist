import os
import sys
import numpy as np
import cv2
import json
from common import find_mxnet
import mxnet as mx
import pandas as pd
import glob
from scipy.stats import gmean
from sklearn.metrics import accuracy_score


def get_d():
    ann_file = 'data/jsons/test2017.json'
    print('Loading annotations from: ' + os.path.basename(ann_file))
    with open(ann_file) as data_file:
        ann_data = json.load(data_file)

    imgs = [aa['file_name'] for aa in ann_data['images']]
    im_ids = [aa['id'] for aa in ann_data['images']]

    print(imgs[:10])
    print(im_ids[:10])
    return dict(zip(imgs, im_ids))


def main():
    d = get_d()
    df = pd.read_csv('data/test.lst', sep='\t', header=None, names=['0', '1', 'fns'])
    print(df.head(10))
    ids = [d.get(fn) for fn in df['fns'].tolist()]

    print(ids[:10])
    print(len(ids))

    y_prd = np.load('tmp/f_avg.npy')
    print(y_prd.shape)

    out = []
    for i in range(len(y_prd)):
        w = y_prd[i]
        idx = np.argsort(w)[::-1]
        out.append(' '.join([str(z) for z in idx[:5]]))

    sb = pd.DataFrame()
    sb['id'] = ids
    sb['predicted'] = out
    sb.sort_values(['id'], inplace=True)
    sb.to_csv('subm/sub3.csv', index=False)


def avg():
    out = np.load('/home/aakuzin/servers/devbox/media/devbox/storage3/tmp/iNaturalist/tmp/rnx101t_22_0.npy')

    for fn in glob.glob('/home/aakuzin/servers/devbox/media/devbox/storage3/tmp/iNaturalist/tmp/*.npy')[1:]:
        tmp = np.load(fn)
        out = gmean(np.array([out, tmp]), axis=0)

    # out = gmean(np.array(out), axis=0)
    np.save('tmp/avg0', out)


def check():
    y_tru = pd.read_csv('data/test.lst', sep='\t', header=None, names=['0', 'y', 'fns'])['y']

    all = []
    # for fn in glob.glob('tmp/r*npy'):
    for fn in ['tmp/rnx101_val_33_0.npy', 'tmp/rnx101t_val_44_0.npy',
               'tmp/rn152k_val_26_0.npy', 'tmp/rnx101t_r_val_44_0.npy']:
        prd = np.load(fn)
        all.append(prd)

        print(accuracy_score(y_tru, np.argmax(prd, axis=1)), fn)

    all = gmean(np.array(all), axis=0)
    print(accuracy_score(y_tru, np.argmax(all, axis=1)), fn)


def agregate():
    y_tru = pd.read_csv('data/test.lst', sep='\t', header=None, names=['0', 'y', 'fns'])['y']

    all = []
    for i in range(0, len(y_tru), 5000):
        fn = 'tmp/320_66_%d.npy' % i
        print(fn)
        #
        prd = gmean(np.load(fn), axis=0)
        print(prd.shape)

        all.append(prd)

    all = np.vstack(all)
    print(all.shape)
    np.save('tmp/f_avg', all)


if __name__ == '__main__':
    # avg()

    # check()
    agregate()
    main()
    # df = pd.read_csv('subm/sub0.csv')
    # df.sort_values(['id'], inplace=True)
    # df.to_csv('subm/sub0.csv', index=False)
