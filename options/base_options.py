import argparse
import os
import shutil

import torch

# import models
# import data


class BaseOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch',
                            type=str,
                            default='res152',
                            help='architecture for binary classification')
        parser.add_argument('--model_name',
                            type=str,
                            default='3branch-cbam',
                            help='model for binary classification',
                            choices=[
                                '3branch', '2branch', '3branch-cbam', '3branch-rcbam',
                                '2branch-cbam', '1branch', '1branch-roi'
                            ])
        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        # parser.add_argument('--blur_prob', type=float, default=0.3)
        # parser.add_argument('--blur_sig', default='0.5')
        # parser.add_argument('--jpg_prob', type=float, default=0.0)
        # parser.add_argument('--jpg_method', default='cv2')
        # parser.add_argument('--jpg_qual', default='75')
        parser.add_argument('--aug_prob', default=0.3, type=float)

        parser.add_argument(
            '--dataroot',
            default='./dataset',
            help=
            'path to images (should have subfolders trainA, trainB, valA, valB, etc)'
        )

        parser.add_argument('--class_bal', action='store_true')
        parser.add_argument('--batch_size',
                            type=int,
                            default=64,
                            help='input batch size')
        parser.add_argument('--loadSize',
                            type=int,
                            default=299,
                            help='scale images to this size')
        parser.add_argument('--roiSize',
                            type=int,
                            default=299,
                            help='scale roi to this size')
        parser.add_argument('--gpu_ids',
                            type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument(
            '--name',
            type=str,
            default='experiment_name',
            help=
            'name of the experiment. It decides where to store samples and models'
        )
        parser.add_argument(
            '--epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model'
        )
        parser.add_argument('--num_threads',
                            default=12,
                            type=int,
                            help='# threads for loading data')
        parser.add_argument('--checkpoints_dir',
                            type=str,
                            default='./checkpoints',
                            help='models are saved here')
        parser.add_argument(
            '--serial_batches',
            action='store_true',
            help=
            'if true, takes images in order to make batches, otherwise takes them randomly'
        )
        # parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop',
        #                     help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        # parser.add_argument('--no_flip', action='store_true',
        #                     help='if specified, do not flip the images for data augmentation')
        parser.add_argument(
            '--init_type',
            type=str,
            default='xavier',
            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument(
            '--init_gain',
            type=float,
            default=0.02,
            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument(
            '--suffix',
            default='',
            type=str,
            help=
            'customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}'
        )
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt, mode):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if not opt.isEval:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        else:
            expr_dir = os.path.join(opt.result_path, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, f'{mode}_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            print(f"Option file saved to {file_name}")

    def parse(self, print_options=True, mode='train'):

        opt = self.gather_options()

        opt.isTrain = self.isTrain  # train or test

        opt.isEval = self.isEval
        # process opt.suffix
        if opt.suffix:
            suffix = (
                '_' +
                opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt, mode)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            pass
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids[0])

        # # additional
        # # opt.classes = opt.classes.split(',')
        # opt.rz_interp = opt.rz_interp.split(',')
        # opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        # opt.jpg_method = opt.jpg_method.split(',')
        # opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        # if len(opt.jpg_qual) == 2:
        #     opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        # elif len(opt.jpg_qual) > 2:
        #     raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
