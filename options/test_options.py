from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path',
                            default = 'checkpoints/1_resnet152/model_epoch_best.pth',
                            choices = ['checkpoints/1-resnet152rcbam-3b-3cls-f12/model_epoch_best.pth', 
                                       'checkpoints/1-resnet152rcbam-3b-f12/model_epoch_best.pth',
                            'checkpoints/1-resnet152rcbam-3b-3cls-f12-wl/model_epoch_best.pth',
                                'checkpoints/1-resnet152rcbam-3b-3cls-nofreeze-2nd/model_epoch_best.pth',
                                       'checkpoints/1-resnet152rcbam-3b-3cls-nofreeze-weighted-loss/model_epoch_best.pth',
                                       'checkpoints/1_resnet152/model_epoch_best.pth',
                                      'checkpoints/1-resnet152cbam-2b/model_epoch_best.pth',
                                      'checkpoints/1-resnet152cbam-3b/model_epoch_best.pth',
                                      'checkpoints/1-resnet152cbam-3b-3cls/model_epoch_best.pth',
                                      'checkpoints/1-resnet152cbam-3b-3cls/model_epoch_latest.pth',
                                      'checkpoints/3_resnet/model_epoch_best.pth'])
                            
        # parser.add_argument('--no_resize', action='store_true')
        # parser.add_argument('--no_crop', action='store_true')
        # parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--testsets', default=[],
                            nargs='+')  # multiple testsets
        parser.add_argument('--result_path', default='./results/')
        parser.add_argument('--test_threshold', default=0.5, type=float)
        parser.add_argument('--isRecord', action='store_true')
        self.isTrain = False
        self.data_aug = False
        self.isEval = True
        return parser
