from torch import nn, Tensor


class IdentityHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
