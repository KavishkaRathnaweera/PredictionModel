import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

use_cuda = 1
class VAE(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        for i in range(1, len(config)):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i - 1], config[i]),
                    nn.ReLU()
                )
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, config[-1])

        for i in range(len(config) - 1, 1, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i], config[i - 1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(config[1], config[0]),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    def decode(self, x):
        result = self.decoder(x)
        return result

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar


def train(data):
    features = data.shape[1]
    data['y'] = data['close']

    x = data.iloc[:, :features].values
    y = data.iloc[:, features].values

    split = int(data.shape[0] * 0.1)
    train_x, test_x = x[: split, :], x[split:, :]
    train_y, test_y = y[: split, ], y[split:, ]

    print(f'trainX: {train_x.shape} trainY: {train_y.shape}')
    print(f'testX: {test_x.shape} testY: {test_y.shape}')

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)

    train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
    test_y = y_scaler.transform(test_y.reshape(-1, 1))

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float()), batch_size=128, shuffle=False)
    # Create dataset = (53,128,41) : num of batches, batch size, features
    # 53 = 6700/128
    model = VAE([features, 400, 400, 400, 10], 10)

    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

    num_epochs = 50
    learning_rate = 0.00003
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    hist = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        total_loss = 0
        loss_ = []
        for (x,) in train_loader:
            x = x.to(device)
            output, z, mu, logVar = model(x)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(output, x) + kl_divergence
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
        hist[epoch] = sum(loss_)
        print('[{}/{}] Loss:'.format(epoch + 1, num_epochs), sum(loss_))

    torch.save(model.state_dict(), '../VAE_model')
    model.eval()
    _, VAE_train_x, train_x_mu, train_x_var = model(torch.from_numpy(train_x).float().to(device))
    _, VAE_test_x, test_x_mu, test_x_var = model(torch.from_numpy(test_x).float().to(device))

    df_tr = pd.DataFrame(VAE_train_x.detach().numpy(),
                         columns=['enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'enc7', 'enc8', 'enc9', 'enc10'])
    df_ts = pd.DataFrame(VAE_test_x.detach().numpy(),
                         columns=['enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'enc7', 'enc8', 'enc9', 'enc10'])
    df_ts.index += train_x_mu.shape[0]
    df_vae = pd.concat([df_tr, df_ts], axis=0)


    return df_vae

def predict(data):
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    features = data.shape[1]
    model = VAE([features, 400, 400, 400, 10], 10)
    model.load_state_dict(torch.load('../VAE_model'))
    model.eval()
    _, VAE_train_x, train_x_mu, train_x_var = model(torch.from_numpy(data).float().to(device))
    df_tr = pd.DataFrame(VAE_train_x.detach().numpy(),
                         columns=['enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'enc7', 'enc8', 'enc9', 'enc10'])
    return df_tr