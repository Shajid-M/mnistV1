from torch import nn, Tensor


class MNISTV1(nn.Module):
    def __init__(self, backbone: nn.Module, adapter: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, x: Tensor):
        x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.adapter(x)
        x = self.head(x)
        return x
