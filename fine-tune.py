import os
import argparse
import logging

from common import data, fit
import mxnet as mx

log_file = "avitonet4k_resnext-101-64x4d_log"
log_dir = "./"
log_file_full_name = os.path.join(log_dir, log_file)
head = '%(asctime)-15s Node[' + str(mx.kvstore.create("local").rank) + '] %(message)s'

logger = logging.getLogger()
handler = logging.FileHandler(log_file_full_name)
formatter = logging.Formatter(head)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import os

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)

    parser.set_defaults(image_shape='3,224,224', num_epochs=30,
                        lr=.01, lr_step_epochs='10,20', wd=0.0001, mom=0.9)

    args = parser.parse_args()

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prefix = 'model/resnext101'
    epoch = 1
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # train
    fit.fit(args=args,
            network=sym,
            data_loader=data.get_rec_iter,
            arg_params=arg_params,
            aux_params=aux_params)
