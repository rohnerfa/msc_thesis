import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils.test_mse import test_mse
from utils.common import NN
from utils.nest1 import NestNet
from utils.nest2 import NestNet2

class MLP(torch.nn.Module):

    def __init__(self, n_width, n_layers, input_dim, output_dim, seed, **kwargs):
        super(MLP, self).__init__()
        self.n_width = n_width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.activation = torch.nn.ReLU()
        self.input_layer = torch.nn.Linear(self.input_dim, self.n_width)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.n_width, self.n_width) for i in range(self.n_layers)])
        self.output_layer = torch.nn.Linear(self.n_width, self.output_dim)
        
        self.apply(self._init_weights)

        
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


class HeightNet:
    def __init__(self, epochs, batch_size_train, learning_rate, n_width, n_layers, n_height, input_dim, output_dim, train_set, test_set, path, name, seed=0, **kwargs):
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.learning_rate = learning_rate
        self.seed = seed
        
        self.n_width = n_width
        self.n_layers = n_layers
        self.n_height = n_height
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.path = path
        self.name = name
        
        self.train_set = train_set
        self.test_set = test_set
        self.g = torch.Generator()
        self.g.manual_seed(0)
        
        self.train_dl = torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size_train,
            worker_init_fn=0,
            generator = self.g)

        self.test_dl = torch.utils.data.DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=len(self.test_set))
        
        if 'val_set' in kwargs:
            self.val_set = kwargs['val_set']
            self.val_dl = torch.utils.data.DataLoader(
                self.val_set,
                shuffle=False,
                batch_size=len(self.val_set))
            
        
        
        if self.n_height == 2:
            self.network = NestNet(n_width=self.n_width, n_layers=self.n_layers, input_dim=self.input_dim, output_dim=self.output_dim, seed=self.seed)
        elif self.n_height == 3:
            self.network = NestNet2(n_width=self.n_width, n_layers=self.n_layers, input_dim=self.input_dim, output_dim=self.output_dim, seed=self.seed)
        else:
            self.network = MLP(n_width=self.n_width, n_layers=self.n_layers, input_dim=self.input_dim, output_dim=self.output_dim, seed=self.seed)

        self.model = NN(self.network, self.learning_rate)
        self.logger = TensorBoardLogger(self.path, name=self.name)
        self.trainer = pl.Trainer(max_epochs=self.epochs, default_root_dir=self.path+'/'+self.name, deterministic=True, auto_lr_find=True, logger=self.logger)
        
        self.training_time = None
        self.test_error = None

    def tune(self):
        self.trainer.tune(self.model, self.train_dl)
    
    def train(self):
        t_0 = time.time()
        if hasattr(self, 'val_dl'):
            self.trainer.fit(model=self.model, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl)
        else:
            self.trainer.fit(model=self.model, train_dataloaders=self.train_dl)
        self.training_time = time.time() - t_0 
    
    def test(self):
        self.test_error = test_mse(self.model, self.test_dl, p=2)
