from ovae import OVAE

# from dataset import SmallPatchDataset

import torch
import pyro
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO

pyro.enable_validation(True)
pyro.clear_param_store()

data = torch.load('datasets/dataset02.pt').cpu()

ova = OVAE()
svi = SVI(ova.model, ova.guide, optim.Adam({'lr': 1e-3, 'amsgrad': True}), loss=TraceGraph_ELBO())

if __name__ == '__main__':
    num_step = 4000
    for i in range(num_step):
        loss = svi.step(data)
        print('i={}, elbo={:.2f}'.format(i, loss))
    # torch.save(ova.encode.state_dict(), 'saved_model/full_color7.pt')
