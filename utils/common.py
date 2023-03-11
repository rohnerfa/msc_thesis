import torch
import torch.optim as optim
import pytorch_lightning as pl
from tqdm.auto import tqdm

def test_mse(network, test_dataloader):
    criterion = torch.nn.MSELoss()
    running_loss = 0.0

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            inputs, labels = data
            outputs = network(inputs).view(-1,)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(test_dataloader)

'''
class MLP(torch.nn.Module):

    def __init__(self, n_width, n_layers):
        super(MLP, self).__init__()
        self.n_width = n_width
        self.n_layers = n_layers
        self.activation = torch.nn.ReLU()
        self.input_layer = torch.nn.Linear(2, self.n_width)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.n_width, self.n_width) for i in range(self.n_layers)])
        self.output_layer = torch.nn.Linear(self.n_width, 1)
        
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
            torch.manual_seed(42)
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                    m.bias.data.zero_()
'''


class NN(pl.LightningModule):
    def __init__(self, network, learning_rate):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.network(x).view(-1,)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.network(x).view(-1,)
        val_loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.network(x).view(-1,)
        test_loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    
        return [optimizer], [{"scheduler": scheduler}]
        #return optimizer