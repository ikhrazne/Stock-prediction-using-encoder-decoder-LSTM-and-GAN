import faulthandler

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from preprocess import Preprocessor


class FirstModel(nn.Module):

    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 batch_size,
                 num_layers=1):
        super(FirstModel, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.dense1 = nn.Linear(hidden_size, 100)
        self.dense2 = nn.Linear(100, 5)
        # self.dense3 = nn.Linear(5, 5)

    def forward(self, x):
        x = x.reshape((x.shape[1], self.seq_len, self.input_size))

        out, (hidden, _) = self.lstm(x)
        out = torch.relu(hidden[-1])
        out = torch.relu(self.dense1(out))
        out = self.dense2(out)
        return out


if __name__ == "__main__":

    NUM_WEEKS = 1
    UNIVARIATE = True
    BATCH_SIZE = 225
    EPOCHS = 200

    SEQ_NUM = NUM_WEEKS * 5
    if UNIVARIATE:
        INPUT_SIZE = 1
    else:
        INPUT_SIZE = 5

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    faulthandler.enable()
    p = Preprocessor(path=r'dax.csv', start_data=2, end_data=-4)

    X_train, y_train, X_test, y_test = p.cluster_data_to_train_test(NUM_WEEKS, univariate=UNIVARIATE)

    loader = DataLoader(TensorDataset(X_train, y_train),
                        shuffle=True,
                        batch_size=BATCH_SIZE,
                        num_workers=4)
    model = FirstModel(INPUT_SIZE, SEQ_NUM, 200, BATCH_SIZE).to(device)

    mse = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001)

    epoch_losses = []
    for epoch in range(250):

        losses = []
        count = 0
        for (X_bach, y_batch) in loader:
            optimizer.zero_grad()
            outputs = model(X_bach.unsqueeze(0))
            loss = mse(outputs, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            count += 1

        l = sum(losses) / len(losses)
        epoch_losses.append(l)
        print(f"Epoch {epoch} loss: {l}")

    outputs = model(X_test.unsqueeze(0))
    loss = mse(outputs, y_test)
    l = sum(epoch_losses) / len(epoch_losses)
    print(f"Train loss: {l}")
    print(f"Test loss: {loss.item()}")
    import matplotlib.pyplot as plt

    plt.plot(epoch_losses)
    plt.show()
