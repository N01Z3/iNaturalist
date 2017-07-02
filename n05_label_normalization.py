import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_balanced():
    df = pd.read_csv('data/train.lst', sep='\t', header=None, names=['0', '1', '2'])
    print(df.head())
    print(df.shape)

    cnt = []
    out = []
    for i in range(5089):
        tdf = df[df['1'] == i]
        if tdf.shape[0] < 500:
            tdf = pd.concat(100 * [tdf], ignore_index=True)
            out.append(tdf[:500])

        else:
            out.append(tdf)

    # plt.plot(sorted(cnt))
    # plt.show()

    df = pd.concat(out, ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.shape)
    df.to_csv('data/train_balanced.lst', sep='\t', header=None)


def check():
    df = pd.read_csv('data/train_balanced.lst', sep='\t', header=None, names=['0', '1', '2'])

    cnt = []
    for i in range(5089):
        tdf = df[df['1'] == i]
        cnt.append(tdf.shape[0])

    plt.plot(sorted(cnt))
    plt.show()


if __name__ == '__main__':
    make_balanced()
    check()