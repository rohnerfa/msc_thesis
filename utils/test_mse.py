import torch
from tqdm import tqdm

def test_mse(network, test_dataloader, p=2):
    if p == 1:
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()
    running_loss = 0.0

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            inputs, labels = data
            outputs = network(inputs).view(-1,)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(test_dataloader)