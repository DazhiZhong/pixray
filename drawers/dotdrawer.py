# this is derived from ClipDraw code
# CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders
# Kevin Frans, L.B. Soros, Olaf Witkowski
# https://arxiv.org/abs/2106.14843

from DrawingInterface import DrawingInterface

import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import PIL.Image

class DotDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--dots", type=int,  default=5000, dest='dots')
        parser.add_argument("--dot_size", type=int, default=3.0, dest='dot_size')
        parser.add_argument("--dot_size_grad", type=bool, default=False, dest='dot_size_grad')
        # parser.add_argument("--min_stroke_width", type=float, help="min width (percent of height)", default=1, dest='min_stroke_width')
        # parser.add_argument("--max_stroke_width", type=float, help="max width (percent of height)", default=5, dest='max_stroke_width')
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()

        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]
        self.dots = settings.dots
        self.dot_size = settings.dot_size
        self.dot_size_grad = settings.dot_size_grad

    def load_model(self, settings, device):
        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        device = torch.device('cuda:0')
        pydiffvg.set_device(device)
        print("clipdraw",device)

        canvas_width, canvas_height = self.canvas_width, self.canvas_height


        # Initialize 
        shapes = []
        shape_groups = []

        radius_vars = []
        center_vars = []
        stroke_width_vars = []
        fill_color_vars = []
        stroke_color_vars = []


        # background shape
        p0 = [0, 0]
        p1 = [canvas_width, canvas_height]
        path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
        shapes.append(path)
        # https://encycolorpedia.com/f2eecb
        cell_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes)-1]), stroke_color = None, fill_color = cell_color)
        shape_groups.append(path_group)

        # path_group.fill_color.requires_grad = True
        fill_color_vars.append(path_group.fill_color)

        #dots
        for i in range(self.dots):
            # num_segments = 1
            # num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            # points = []
            # p0 = (random.random(), random.random())
            # points.append(p0)
            # for j in range(num_segments):
            #     radius = 0.1
            #     p3 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            #     points.append(p3)
            #     p0 = p3
            # points = torch.tensor(points)
            # points[:, 0] *= canvas_width
            # points[:, 1] *= canvas_height

            r = random.random()*canvas_width, random.random()*canvas_height
            r = torch.tensor(r)

            cir = pydiffvg.Circle(torch.tensor(self.dot_size,dtype=torch.float), r, stroke_width=torch.tensor(0.0,dtype=torch.float))
            shapes.append(cir)


            # path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor((min_width + max_width)/4), is_closed = False)
            # shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = torch.tensor([random.random(), random.random(), random.random(),random.random()]), stroke_color = None)
            shape_groups.append(path_group)

        # Just some diffvg setup
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

        for path in shapes[1:]:
            path.radius.requires_grad = True
            radius_vars.append(path.radius)
            path.center.requires_grad = True
            center_vars.append(path.center)
            # path.stroke_width.requires_grad = True
            # stroke_width_vars.append(path.stroke_width)
        for group in shape_groups[1:]:
            group.fill_color.requires_grad = True
            fill_color_vars.append(group.fill_color)
            # group.stroke_color.requires_grad = True
            # stroke_color_vars.append(group.stroke_color)

        self.radius_vars = radius_vars
        self.center_vars = center_vars
        self.stroke_width_vars = stroke_width_vars
        self.fill_color_vars = fill_color_vars
        self.stroke_color_vars = stroke_color_vars
        self.img = img
        self.shapes = shapes 
        self.shape_groups  = shape_groups
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def get_opts(self, decay_divisor):
        # Optimizers
        radius_optim = torch.optim.Adam(self.radius_vars, lr=1.0/decay_divisor)
        center_optim = torch.optim.Adam(self.center_vars, lr=1.0/decay_divisor)
        # width_optim = torch.optim.Adam(self.stroke_width_vars, lr=0.1/decay_divisor)
        
        fill_color_optim = torch.optim.Adam(self.fill_color_vars, lr=0.01/decay_divisor)
        # stroke_color_optim = torch.optim.Adam(self.stroke_color_vars, lr=0.01/decay_divisor)
        # opts = [radius_optim,center_optim,width_optim,fill_color_optim,stroke_color_optim]
        opts = [radius_optim,center_optim,fill_color_optim]
        return opts

    def rand_init(self, toksX, toksY):
        # TODO
        pass

    def init_from_tensor(self, init_tensor):
        # TODO
        pass

    def reapply_from_tensor(self, new_tensor):
        # TODO
        pass

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        return None

    def synth(self, cur_iteration):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = render(self.canvas_width, self.canvas_height, 2, 2, cur_iteration, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        self.img = img
        return img

    @torch.no_grad()
    def to_image(self):
        img = self.img.detach().cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = np.uint8(img * 254)
        # img = np.repeat(img, 4, axis=0)
        # img = np.repeat(img, 4, axis=1)
        pimg = PIL.Image.fromarray(img, mode="RGB")
        return pimg

    def clip_z(self):
        with torch.no_grad():
            for group in self.shape_groups:
                group.fill_color.data[3].clamp_(0.0, 1.0)
                # group.stroke_color.data[3].clamp_(0.0, 0.0)
        # with torch.no_grad():
            # for path in self.shapes:
            #     path.stroke_width.data.clamp_(self.min_width, self.max_width)
            # for group in self.shape_groups:
            #     group.fill_color.data.clamp_(0.0, 1.0)

    def get_z(self):
        return None

    def get_z_copy(self):
        return None

    def set_z(self, new_z):
        return None

