"""
Takes latent code and generate the (latent variable) of object locations
\TODO try either Bernoulli Variable or Bernoulli parameter as latent variable
"""

import torch
import pyro
import pyro.distributions as dist


# poisson process
class ExactlyPoisson:

    def __init__(self, max_num=4):
        # maximum number is necessary as we can not sample different dimensions in mini-batches
        self.num = max_num
        # the coordinates are scaled to [-1,1], however incomplete object may appear on the edge
        self.prior_low = torch.tensor([-1.0, -1.0])
        self.prior_high = torch.tensor([1.0, 1.0])

    def sample_prior(self, n):
        z_e = pyro.sample('z_e', dist.Bernoulli(torch.tensor([0.5]).expand(n, self.num)).to_event(1))
        z_x = pyro.sample('z_x', dist.Uniform(self.prior_low[0].expand(n, self.num),
                                              self.prior_high[0].expand(n, self.num)).mask(z_e).to_event(1))
        z_y = pyro.sample('z_y', dist.Uniform(self.prior_low[1].expand(n, self.num),
                                              self.prior_high[1].expand(n, self.num)).mask(z_e).to_event(1))
        z_l = torch.stack((z_x, z_y), 2).unsqueeze(-1)

        return z_l, z_e
