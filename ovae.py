"""
Object-based Variational Auto-encoder
"""

import math
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from point_sub import ExactlyPoisson
from object_sub import EllipseObject
from modules import SPEncoder


class OVAE(nn.Module):
    def __init__(self, size=31,
                 num=4,
                 point_generator=ExactlyPoisson,
                 object_generator=EllipseObject,
                 sd=0.03,
                 use_cuda=False):

        super(OVAE, self).__init__()

        if use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.size = size
        self.nop = num
        self.point_generator = point_generator(self.nop)
        self.object_generator = object_generator()
        self.encode = SPEncoder()
        # self.encode.load_state_dict(torch.load('saved_model/full_color7.pt'))
        self.sd = torch.tensor(sd)

    def model(self, data):
        with pyro.plate("data", data.shape[0]) as idx:
            batch = data[idx]
            n = batch.size(0)
            z_l, z_e = self.point_generator.sample_prior(n)
            rx, ry, theta, nuc_col, bkg_col = self.object_generator.sample_prior(z_e)
            img = self.object_generator.generate_img(self.size, z_l, z_e, rx, ry, theta, nuc_col, bkg_col)
            pyro.sample('obs', dist.Normal(img.view(n, -1), self.sd.expand_as(img).view(n, -1)).to_event(1),
                        obs=batch.view(n, -1))

    def guide(self, data):
        # amortized MAP guide
        pyro.module('encode', self.encode)
        with pyro.plate("data", data.shape[0], subsample_size=32) as idx:
            batch = data[idx]
            features = self.encode(batch)
            # cec = features[:, :self.nop]
            cec = torch.ones((batch.shape[0], self.nop))
            pyro.sample('z_e', dist.Bernoulli(cec).to_event(1))
            z_x = 2*features[:, self.nop:2*self.nop]-1
            z_y = 2*features[:, 2*self.nop:3*self.nop]-1
            pyro.sample('z_x', dist.Delta(z_x).to_event(1))
            pyro.sample('z_y', dist.Delta(z_y).to_event(1))
            rx = (features[:, 3*self.nop:4*self.nop]) + 1e-5
            ry = (features[:, 4*self.nop:5*self.nop]) + 1e-5
            theta = features[:, 5*self.nop:6*self.nop]*0.5*math.pi
            pyro.sample('rx', dist.Delta(rx).to_event(1))
            pyro.sample('ry', dist.Delta(ry).to_event(1))
            pyro.sample('theta', dist.Delta(theta).to_event(1))
            nuc_col = features[:, 6*self.nop:9*self.nop]
            pyro.sample('nuc_col', dist.Delta(nuc_col).to_event(1))
            bkg_col = features[:, 9*self.nop:]
            pyro.sample('bkg_col', dist.Delta(bkg_col).to_event(1))
