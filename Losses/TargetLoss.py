import torch
from urllib.request import urlopen
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF
from Losses.LossInterface import LossInterface
import lpips

class TargetLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.t_im = None
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        if self.device!="cpu":
            self.loss_fn_vgg.cuda()
        
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("-tti",  "--true_target_image", type=str, help="this gives a mse loss for a given image, hard way", default="", dest='true_target_image')
        parser.add_argument("-ttw",  "--true_target_weight", type=float, help="true_target_image loss weights", default=0, dest='true_target_weight')
        parser.add_argument("--true_target_multi", type=bool, help="use l1 and l2", default=False, dest='true_target_multi')
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
        target_im  = TF.resize(target_im, (args.size[0],args.size[1]),TF.InterpolationMode.BICUBIC)
        print('resize?')
        lossglobals = {
            "target_im":target_im,
        }
        return lossglobals

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        if self.t_im is None:
            self.t_im = TF.resize(lossGlobals['target_im'], out.size()[2:4],TF.InterpolationMode.BICUBIC)
        cur_loss = torch.sum(self.loss_fn_vgg(out, self.t_im)).to(self.device)
        if args.true_target_multi:
            l1 = nn.L1Loss()
            mse = nn.MSELoss()
            cur_loss+=l1(out, self.t_im).to(self.device)*0.5
            cur_loss+=mse(out, self.t_im).to(self.device)*0.5     
        return cur_loss * args.true_target_weight


