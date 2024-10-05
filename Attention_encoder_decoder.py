import faulthandler

import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from attention import Attention
from torch.utils.data import DataLoader, TensorDataset

from preprocess import Preprocessor


class DecoderAttention(nn.Module):

    def __init__(self, input_size, seq_len, hidden_size):
        super(DecoderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=200 + input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=1)
        self.tdd1 = nn.Linear(hidden_size, 100)
        self.tdd2 = nn.Linear(100, 50)
        self.tdd3 = nn.Linear(50, 1)
        self.attention = Attention(hidden_size)

    def forward_step(self, x, hidden_state, encoder_output):
        x = x.reshape((x.shape[0], self.seq_len, self.input_size))
        query = hidden_state[0].permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_output)
        input_lstm = torch.cat((x, context), dim=2)
        # print(input_lstm.shape)
        out, (hidden, _) = self.lstm(input_lstm, hidden_state)
        out = torch.relu(hidden[-1])
        out = torch.relu(self.tdd1(out))
        out = torch.relu(self.tdd2(out))
        out = self.tdd3(out)
        return out

    def forward(self, x, encoded_hidden, target_tensor=None, encoder_output=None):

        decoded_input = x

        outputs = []
        attentions = []

        for i in range(5):
            decoded_output = self.forward_step(decoded_input, encoded_hidden, encoder_output)
            outputs.append(decoded_output)
            if target_tensor is None:
                # teacher forcing
                decoded_input = target_tensor[:, i].unsqueeze(1)
            else:
                # no teacher forcing
                decoded_input = decoded_output

        decoded_outputs = torch.cat(outputs, 1)
        return decoded_outputs


class EncoderDecoderLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, seq_len=5):
        super(EncoderDecoderLSTM, self).__init__()
        self.encoder = Encoder(input_size, seq_len, hidden_size)
        self.decoder = DecoderAttention(1, 1, hidden_size)

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
                # print(X_batch)
                encoded_out, hidden = self.encoder(X_batch.unsqueeze(0))
                decoded_input = torch.tensor([[x] for x in X_batch[:, -1]])
                outputs = self.decoder(decoded_input, hidden, y_batch, encoded_out)

                loss = criterion(outputs, y_batch.unsqueeze(0))
                # loss.requires_grad = True
                loss.backward()
                optimizer.step()
                # print(f"Epoch {epoch} loss: {loss.item()}")
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

    NUM_WEEKS = 5
    UNIVARIATE = True
    SEQ_NUM = NUM_WEEKS * 5
    BATCH_SIZE = 25
    EPOCHS = 200


    if UNIVARIATE:
        INPUT_SIZE = 1
    else:
        INPUT_SIZE = 5

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    p = Preprocessor(path='dax.csv', start_data=2, end_data=-4, sep=";")
    X_train, y_train, X_test, y_test = p.cluster_data_to_train_test(NUM_WEEKS, univariate=UNIVARIATE)

    model = EncoderDecoderLSTM(INPUT_SIZE, 200, SEQ_NUM).to(device)

    model.train_model(X_train, y_train, X_test, y_test, BATCH_SIZE, EPOCHS)

