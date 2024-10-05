import faulthandler

import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from decoder import Decoder
from torch.utils.data import DataLoader, TensorDataset

from preprocess import Preprocessor


class EncoderDecoderLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderDecoderLSTM, self).__init__()
        self.encoder = Encoder(input_size, 5, hidden_size)
        self.decoder = Decoder(1, 1, hidden_size)

    def forward(self, x):
        encoded_out, hidden = self.encoder(x)
        outputs = self.decoder(x, hidden)
        return outputs

    def train_model(
            self,
            input_tensor,
            target_tensor,
            test_input_tensor,
            test_target_tensor,
            batch_size,
            epochs
    ):

        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # optimizer = optim.SGD(self.parameters(),
        #                      lr=0.001,
        #                      momentum=0.9)

        dataloader = DataLoader(TensorDataset(input_tensor, target_tensor),
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4)

        epoch_losses = []

        for epoch in range(epochs):
            losses = []
            for (X_batch, y_batch) in dataloader:
                optimizer.zero_grad()
                encoded_out, hidden = self.encoder(X_batch.unsqueeze(0))
                decoded_input = torch.tensor([[x] for x in X_batch[:, -1]])
                outputs = self.decoder(decoded_input, hidden, y_batch)
                loss = criterion(outputs, y_batch.unsqueeze(0))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            l = sum(losses) / len(losses)
            epoch_losses.append(l)
            print(f"Epoch {epoch} loss: {l}")

        print(f"train loss : {epoch_losses[-1]}")
        test_outputs = self.forward(test_input_tensor)
        test_loss = criterion(test_outputs, test_target_tensor)
        print(f"test loss: {test_loss.item()}")

        import matplotlib.pyplot as plt
        plt.plot(epoch_losses)
        plt.show()


if __name__ == "__main__":
    faulthandler.enable()

    NUM_WEEKS = 1
    UNIVARIATE = True
    BATCH_SIZE = 225
    EPOCHS = 250

    SEQ_NUM = NUM_WEEKS * 5
    if UNIVARIATE:
        INPUT_SIZE = 1
    else:
        INPUT_SIZE = 5

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    p = Preprocessor(path='dax.csv', start_data=2, end_data=-4, sep=";")
    X_train, y_train, X_test, y_test = p.cluster_data_to_train_test(NUM_WEEKS)

    model = EncoderDecoderLSTM(INPUT_SIZE, 200).to(device)

    model.train_model(X_train, y_train, X_test, y_test, BATCH_SIZE, EPOCHS)
