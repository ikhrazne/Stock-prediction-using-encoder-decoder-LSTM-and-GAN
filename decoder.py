
import torch
import torch.nn as nn
import math
from timeDistributed import TimeDistributed


class Decoder(nn.Module):

    def __init__(self, input_size, seq_len, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=1)
        self.tdd1 = nn.Linear(hidden_size, 100)
        self.tdd2 = nn.Linear(100, 50)
        self.tdd3 = nn.Linear(50, 1)

    def forward_step(self, x, hidden_state):
        x = x.reshape((x.shape[0], self.seq_len, self.input_size))
        out, (hidden, _) = self.lstm(x, hidden_state)
        out = torch.relu(hidden[-1])
        out = torch.relu(self.tdd1(out))
        out = torch.relu(self.tdd2(out))
        out = self.tdd3(out)
        return out

    def forward(self, x, encoded_hidden, target_tensor=None):

        decoded_input = x

        outputs = []

        for i in range(5):
            decoded_output = self.forward_step(decoded_input, encoded_hidden)
            outputs.append(decoded_output)
            if target_tensor is not None:
                # teacher forcing
                decoded_input = target_tensor[:, i].unsqueeze(1)
            else:
                # no teacher forcing
                decoded_input = decoded_output

        decoded_outputs = torch.cat(outputs, 1)
        return decoded_outputs