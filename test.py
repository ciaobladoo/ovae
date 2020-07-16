from ovae import OVAE
from modules import SPEncoder

import torch
import math
import matplotlib.pyplot as plt

model = SPEncoder()
model.load_state_dict(torch.load('saved_model/two_ellipse.pt'))
model.eval()

img = torch.load('datasets/testdata04.pt')
features = model(img)

self = OVAE()

z_x = 2*features[:, self.nop:2*self.nop]-1
z_y = 2*features[:, 2*self.nop:3*self.nop]-1
z_l = torch.stack((z_x, z_y), 2).unsqueeze(-1)
rx = (features[:, 3*self.nop:4*self.nop]) + 1e-5
ry = (features[:, 4*self.nop:5*self.nop]) + 1e-5
theta = features[:, 5*self.nop:6*self.nop]*0.5*math.pi
nuc_col = features[:, 6*self.nop:9*self.nop]
bkg_col = features[:, 9*self.nop:]
z_e = torch.ones(img.size(0), self.nop)

img = self.object_generator.generate_img(self.size, z_l, z_e, rx, ry, theta, nuc_col, bkg_col).detach().cpu()
for i in range(8):
    plt.imshow(img[i].permute(1,2,0))
    plt.show()
