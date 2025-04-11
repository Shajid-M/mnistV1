from torch import nn, Tensor, flatten


class LinearAdapter(nn.Module):
    def __init__(self, in_features, out_features, flatten_input):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.flatten_inp = flatten_input

        self.linear = nn.Linear(in_features = self.in_features,
                                out_features = self.out_features)
        
    def forward(self, x):
        if self.flatten_inp:
            x = flatten(x, 1)
        return self.linear(x)