import torch
from utils.act1 import Height1Net

class NestNet2(torch.nn.Module):
    def __init__(self, n_width, n_layers, input_dim, output_dim, seed):
        super(NestNet2, self).__init__()
        self.n_width = n_width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        
        self.input_layer = torch.nn.Linear(self.input_dim, self.n_width)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.n_width, self.n_width) for i in range(self.n_layers)])
        self.output_layer = torch.nn.Linear(self.n_width, self.output_dim)
        self.apply(self._init_weights)
        self.activation = Height1Net(self.n_width)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        for i, l in enumerate(self.linears):
            x = self.linears[i](x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.manual_seed(self.seed)
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                    m.bias.data.zero_()