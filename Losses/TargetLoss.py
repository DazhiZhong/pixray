import torch
from urllib.request import urlopen
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF
from Losses.LossInterface import LossInterface

class TargetLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("-tti",  "--true_target_image", type=str, help="this gives a mse loss for a given image, hard way", default="", dest='true_target_image')
        parser.add_argument("-ttw",  "--true_target_weight", type=float, help="true_target_image loss weights", default=0, dest='true_target_weight')
        return parser

    def parse_settings(self,args):
        if (args.true_target_image and ((type(args.true_target_image)==list) or (type(args.true_target_image)==tuple))):
            args.true_target_image = args.true_target_image[0]
        return args

    def add_globals(self, args):
        if args.true_target_image:
            if 'http' in args.true_target_image:
                target_im = Image.open(urlopen(args.true_target_image ))
            else:
                target_im = Image.open(args.true_target_image )
        target_im = TF.to_tensor(target_im)
        target_im = target_im.to(self.device)
        lossglobals = {
            "target_im":target_im,
        }
        return lossglobals

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        mseloss = nn.MSELoss()
        cur_loss = mseloss(out, lossGlobals['target_im']) 
        return cur_loss * args.true_target_weight
