import torch
from torch.nn import functional as F
from Losses.LossInterface import LossInterface
import kornia

class CannyLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--canny_weight", type=float, default=1.0, dest='canny_weight')
        return parser
   
    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        canny_out = kornia.filters.canny(out)
        cur_loss = F.mse_loss(out, torch.flip(out,[3]).detach())
        return cur_loss * args.canny_weight
