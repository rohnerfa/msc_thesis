import torch

class LearnedActivation(torch.nn.Module):

    def __init__(self):
        super(LearnedActivation, self).__init__()
        
        self.activation = torch.nn.ReLU()
        self.input_layer = torch.nn.Linear(1,3)
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
        x = self.output_layer(x)
        return x.view(-1,size)
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.manual_seed(0)
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                    m.bias.data.zero_()


class Height1Net(torch.nn.Module):

    def __init__(self, n_width):
        super(Height1Net, self).__init__()
        
        self.input_layer = torch.nn.Linear(1,3)
        self.linear = torch.nn.Linear(3,3)
        self.output_layer = torch.nn.Linear(3,1)
        
        self.size = n_width
        self.apply(self._init_weights)
        #self.input_layer.weight = torch.nn.Parameter(torch.tensor([1., 1., 1.]).view(-1, 1))
        #self.input_layer.bias = torch.nn.Parameter(torch.tensor([-0.2, -0.1, 0.]))
        #self.output_layer.weight = torch.nn.Parameter(torch.tensor([1., 1., -1.]).view(1, -1))
        #self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))
        #self.linear.weight = torch.nn.Parameter(torch.tensor([[0., 0, 0], [0,  0, 0], [0.2, -0.3, -0.5]]))
        #self.linear.bias = torch.nn.Parameter(torch.tensor([-0.1, 0.5, 0.5]))
        self.activation = LearnedActivation()

    def forward(self, x):
        x = self.input_layer(x.view(-1,1))
        x = self.activation(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x.view(-1,self.size)
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.manual_seed(0)
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                    m.bias.data.zero_()