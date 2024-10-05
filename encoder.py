
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, seq_len, hidden_size, num_layer=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_layer)
        # self.dense = nn.Linear(hidden_size, 5)

    def forward(self, x):
        x = x.reshape((x.shape[1], self.seq_len, self.input_size))
        out, hidden = self.lstm(x)
        return out, hidden
