import torch

class ActivationNet(torch.nn.Module):

    def __init__(self):
        super(ActivationNet, self).__init__()
        
        self.activation = torch.nn.ReLU()
        self.input_layer = torch.nn.Linear(1,3)
        #self.linear = torch.nn.Linear(3,3)
        self.output_layer = torch.nn.Linear(3,1)

        #self.apply(self._init_weights)
        self.input_layer.weight = torch.nn.Parameter(torch.tensor([1., 1., 1.]).view(-1, 1))
        self.input_layer.bias = torch.nn.Parameter(torch.tensor([-0.2, -0.1, 0.]))
        self.output_layer.weight = torch.nn.Parameter(torch.tensor([1., 1., -1.]).view(1, -1))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        size = x.shape[1]
        x = self.input_layer(x.view(-1,1))
        x = self.activation(x)
        #x = self.linear(x)
        #x = self.activation(x)
        x = self.output_layer(x)
        return x.view(-1,size)
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.manual_seed(0)
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                    m.bias.data.zero_()

class NestNet(torch.nn.Module):
    def __init__(self, n_width, n_layers, input_dim, output_dim, seed, **kwargs):
        super(NestNet, self).__init__()
        self.n_width = n_width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        
        self.input_layer = torch.nn.Linear(self.input_dim, self.n_width)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.n_width, self.n_width) for i in range(self.n_layers)])
        self.output_layer = torch.nn.Linear(self.n_width, self.output_dim)
        self.apply(self._init_weights)
        self.activation = ActivationNet()

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