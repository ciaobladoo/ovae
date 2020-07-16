"""
Take latent code of objects as well as latent variable of locations to generate image
"""

import torch
from torch.nn.functional import grid_sample, affine_grid

import pyro
import pyro.distributions as dist

import math


# Ellipses are generated by affine transformation of unit circle, taking five parameters
class EllipseObject:

    def __init__(self, unit_size=16, rxu=-1.3, rxv=0.1):
        # parameters for log-normal distribution of axis length
        self.rxu = torch.tensor(rxu)
        self.rxv = torch.tensor(rxv)

        # generate original image of unit circle
        x = torch.linspace(0, unit_size-1, unit_size).view(1, -1)
        y = torch.linspace(0, unit_size-1, unit_size).view(-1, 1)
        self.circle = (((x-(unit_size-1)/2)/(unit_size/2))**2 + ((y-(unit_size-1)/2)/(unit_size/2))**2 <= 1).float()

    def sample_prior(self, z_e):
        [n, m] = z_e.size()

        # Sample major, minor-axis length (rx, ry) using log-normal distribution
        rx = pyro.sample('rx', dist.LogNormal(self.rxu.expand(n, m), self.rxv.expand(n, m)).mask(z_e).to_event(1))
        ry = pyro.sample('ry', dist.LogNormal(self.rxu.expand(n, m), self.rxv.expand(n, m)).mask(z_e).to_event(1))
        # Sample rotation angle using uniform distribution
        theta = pyro.sample('theta', dist.Uniform(torch.tensor(-1e-5).expand(n, m),
                                                  torch.tensor(0.5*math.pi+1e-5).expand(n, m)).mask(z_e).to_event(1))
        # sample the prior of RGB using uniform distribution
        nuc_col = pyro.sample('nuc_col', dist.Normal(torch.tensor([0.63, 0.27, 0.24]).repeat(n, m),
                                                      torch.tensor(0.05).expand(n, m*3))
                              .mask(z_e.unsqueeze(-1).repeat(1,1,3).view(n,-1)).to_event(1))
        bkg_col = pyro.sample('bkg_col', dist.Normal(torch.tensor([[0.94, 0.68, 0.51]]).expand(n, -1),
                                                      torch.tensor(0.05).expand(n, 3)).to_event(1))

        return rx, ry, theta, nuc_col, bkg_col

    def generate_img(self, size, z_l, z_e, rx, ry, theta, nuc_col, bkg_col):
        [n, m] = z_e.size()

        # Calculate affine transformation using (cx, cy, rx, ry, theta)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        tav = cos/rx
        tbv = sin/rx
        tcv = sin/ry
        tdv = -cos/ry
        qat = torch.stack((torch.stack((tav, tcv), 2), torch.stack((tbv, tdv), 2)), 3)
        tat = -torch.matmul(qat, z_l)
        afm = torch.cat((qat, tat), -1).view(-1, 2, 3)
        # transform circles to ellipses thus generating sketches of the image
        grid = affine_grid(afm, torch.Size((afm.size(0), 1, size, size)))
        diameter = self.circle.size(0)
        sketches = grid_sample(self.circle.unsqueeze(0).unsqueeze(0).expand(afm.size(0), 1, diameter, diameter), grid)\
            .view(n, m, 1, size, size)  # (N, M, C, size, size)
        sketches = sketches * z_e.unsqueeze(2).unsqueeze(2).unsqueeze(2)  # delete nucleus that does not exist
        # color nucleus
        nbk_img = (sketches * nuc_col.view(n, m, 3).unsqueeze(-1).unsqueeze(-1)).sum(1)
        # add background color
        bkg_ind = torch.ones((n, 1, size, size)) - sketches.sum(1)
        bkg_img = bkg_ind * bkg_col.unsqueeze(-1).unsqueeze(-1)

        img = bkg_img + nbk_img

        return img