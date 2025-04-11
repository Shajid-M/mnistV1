from torch import Tensor, nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        return self.cross_entropy_loss(preds, targets)