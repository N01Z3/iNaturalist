import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.legend_handler import HandlerLine2D

split2 = 'Validation-accuracy='
split = 'alidation-top_k_accuracy_5='
out = []

fls = glob.glob('logs/*log')
print(fls)

for fn, c in zip(fls, ['r', 'g', 'b', 'c', 'k', 'y', 'navy', 'peru']):
    out = []
    with open(fn) as f:
        content = f.readlines()

    content = [x.strip('\n') for x in content]

    for line in content:
        if split in line:
            val = line.split(split)[-1]
            out.append(float(val))
            # print(val.replace('.',','))

    # print(out)
    # print(np.amax(out))
    # print(np.amax(out))
    print(c, fn)

    print(np.array(out)[np.argsort(out)[-3:]], np.argsort(out)[-3:])

    line1, = plt.plot(out, color=c, label='%0.4f: %s' % (np.amax(out), fn.split('_')[-1]))
    plt.ylim(0.6, 0.8)

plt.legend(handler_map={line1: HandlerLine2D(numpoints=3)})
plt.show()
