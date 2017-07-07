import os
import sys
import numpy as np
import cv2
import json
from common import find_mxnet
import mxnet as mx
import pandas as pd

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

    y_prd = np.load('tmp/esnext-101_val_30_0.npy')
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
    sb.to_csv('subm/sub1.csv', index=False)


if __name__ == '__main__':
    main()
    # df = pd.read_csv('subm/sub0.csv')
    # df.sort_values(['id'], inplace=True)
    # df.to_csv('subm/sub0.csv', index=False)