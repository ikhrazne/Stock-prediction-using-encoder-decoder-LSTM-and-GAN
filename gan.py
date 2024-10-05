
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import Preprocessor
from encoder import Encoder
from decoder import Decoder
import faulthandler


class Generator(nn.Module):

    def __init__(
            self,
            input_size,
            seq_len,
            hidden_size,
            num_layers=1
    ):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.encoder = Encoder(input_size, self.seq_len, hidden_size)
        self.decoder = Decoder(input_size, 1, hidden_size)
        # self.lstm = nn.LSTM(input_size=input_size,
        #                    hidden_size=hidden_size,
        #                    batch_first=True,
        #                    num_layers=num_layers
        #                    )
        # self.dense = nn.Linear(hidden_size, 150)
        # self.dense2 = nn.Linear(150, 100)
        # self.dense3 = nn.Linear(100, 5)

    def forward(self, x, target_tensor=None):
        # print(x.shape)
        # x = x.reshape((x.shape[1], self.seq_len, self.input_size))
        #out, (hidden, _) = self.lstm(x)
        #out = torch.relu(hidden[-1])
        #out = torch.relu(self.dense(out))
        #out = self.dense2(out)
        #out = self.dense3(out)
        # print(target_tensor)
        decoded_input = torch.tensor([[x] for x in X_batch[:, -1]])
        encoded_output, hidden = self.encoder(x)
        out = self.decoder(decoded_input, hidden, target_tensor)
        return out


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, 150)
        self.dense2 = nn.Linear(150, 1)

    def forward(self, x):
        out = torch.relu(self.dense(x))
        out = torch.relu(self.dense1(out))
        out = torch.sigmoid(self.dense2(out))
        return out


if __name__ == "__main__":
    faulthandler.enable()

    NUM_WEEKS = 1
    INPUT_SIZE = 1
    SEQ_LEN = 5

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    p = Preprocessor(path='dax.csv', start_data=2, end_data=-4, sep=";")
    X_train, y_train, X_test, y_test = p.cluster_data_to_train_test(NUM_WEEKS)

    generator = Generator(INPUT_SIZE, SEQ_LEN, 200).to(device)
    discriminator = Discriminator(5, 200).to(device)

    loader = DataLoader(TensorDataset(X_train, y_train),
                        shuffle=True,
                        batch_size=50,
                        num_workers=4)

    loss = nn.BCELoss()
    g_loss = nn.MSELoss()

    optimizer_G = optim.Adam(generator.parameters(),
                             lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=0.001)

    epoch_losses = []
    for epoch in range(200):
        losses = []
        for (X_batch, y_batch) in loader:

            # Train the generator

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            fake_outputs = generator(X_batch.unsqueeze(0), y_batch)

            real_values = discriminator(y_batch)

            real_losses = loss(real_values, torch.ones([X_batch.shape[0], 1]))
            # print(fake_outputs.shape)
            fake_values = discriminator(fake_outputs)
            # print(fake_values.shape)
            fake_losses = loss(fake_values, torch.zeros([X_batch.shape[0], 1]))
            d_loss = (real_losses + fake_losses)

            discriminator.zero_grad()

            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            # g_losses = loss(fake_outputs, labels)
            output_fake = discriminator(fake_outputs)
            # print(output_fake)
            g_losses = loss(output_fake, torch.ones([X_batch.shape[0], 1]))
            generator.zero_grad()
            g_losses.backward()
            optimizer_G.step()

            g_outputs = generator(X_batch.unsqueeze(0), y_batch)
            g_second_loss = g_loss(g_outputs, y_batch)

            losses.append(g_second_loss.item())
            # print(f"Epoch {epoch} Generator loss: ", g_second_loss.item())

        l = sum(losses) / len(losses)
        epoch_losses.append(l)
        print(f"Epoch {epoch} loss: {l}")

    import matplotlib.pyplot as plt
    plt.plot(epoch_losses)
    plt.show()
