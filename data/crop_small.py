import pandas as pd

def split_train():
    df = pd.read_csv('train.lst', header=None, sep="\t", names=['0', '1', '2'])
    print(df.head())
    print(df.shape)

    chunk = df.shape[0]//5+1
    print(chunk)
    for i in range(5):
        tdf = df[i*chunk: (i+1)*chunk]
        print(tdf.shape)
        tdf.to_csv('train_pt%d.lst' % i, index=False, header=None, sep='\t')


def split_val():
    df = pd.read_csv('val.lst', header=None, sep="\t", names=['0', '1', '2'])
    lbs = sorted(set(df['1'].tolist()))
    print(lbs[:10])
    print(len(lbs))

    out = [df[df['1'] == y][:3] for y in lbs]

    sdf = pd.concat(out, ignore_index=True)
    print(sdf.head())
    print(sdf.shape)
    sdf.to_csv('val_s.lst', index=False, header=None, sep='\t')


if __name__ == '__main__':
    split_val()