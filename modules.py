import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SPEncoder(nn.Module):
    def __init__(self):
        super(SPEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(256, 21, kernel_size=3, stride=2)
        init.xavier_uniform_(self.conv1.weight.data)
        init.xavier_uniform_(self.conv2.weight.data)
        init.xavier_uniform_(self.conv3.weight.data)
        init.xavier_uniform_(self.conv4.weight.data)
        self.conv1.bias.data.zero_()
        self.conv2.bias.data.zero_()
        self.conv3.bias.data.zero_()
        self.conv4.bias.data.zero_()

    def forward(self, x):
        m = x.mean((2,3), True)
        s = x.view(x.size(0), x.size(1), -1).std(2)
        x = (1/s).unsqueeze(-1).unsqueeze(-1)*(x-m)
        # x = 2*(x-0.5)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x)).squeeze(-1).squeeze(-1)

        return x
