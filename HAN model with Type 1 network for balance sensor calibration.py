import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

torch.manual_seed(7)    # reproducible

train_data = torch.from_numpy(np.load('balance sensor train data.npy')).type(torch.FloatTensor)
train_pressure = torch.from_numpy(np.load('balance sensor train pressure.npy')).type(torch.FloatTensor)
test_data = torch.from_numpy(np.load('balance sensor test data.npy')).type(torch.FloatTensor)
test_pressure = torch.from_numpy(np.load('balance sensor test pressure.npy')).type(torch.FloatTensor)
fake_data = torch.from_numpy(np.load('balance sensor fake data.npy')).type(torch.FloatTensor)
fake_prediction = None

train_dataset = Data.TensorDataset(train_data, train_pressure)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=800, shuffle=True)


def I_P(P, C1, C2):     # do not use P**(2/3) due to some error when P is negative
    return C1*(P**2)**(1/3)-C2*(P**4)**(1/3)


class Regression_net1(nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Regression_net1, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)

    def forward(self, x):
        return torch.max(self.hidden(x), dim=1)[0]


net = Regression_net1(225, 32)
loss_func = torch.nn.MSELoss()

learning_rate = [1e-4, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7]
tolerance = [1e-4, 1e-6, 1e-6, 1e-6, 1e-6, 1e-7]
epoch = 0
plt.ion()
for iteration in range(6):             # train neural network and least square alternately
    print(iteration, epoch)                    # during 1st iteration, neural network parameters are set based on default random initialization
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate[iteration])
    train_loss_before = 1e5
    train_loss = train_loss_before - 1
    while abs(train_loss_before - train_loss) > tolerance[iteration]:
        epoch += 1
        train_loss_before = train_loss
        train_prediction = net(train_data)
        train_loss = loss_func(train_prediction, train_pressure)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        plt.cla()
        plt.plot(train_data.numpy().min(axis=1), train_pressure.numpy(), 'y.')
        plt.plot(train_data.numpy().max(axis=1), train_pressure.numpy(), 'k.')
        plt.xlabel('pixel value')
        plt.ylabel('pressure/kg/cm2')
        plt.grid(True)
        test_prediction = net(test_data)
        test_loss = loss_func(test_prediction, test_pressure)
        # plt.plot(test_data.numpy().max(axis=1), test_prediction.data.numpy(), 'r.')
        fake_prediction = net(fake_data)
        plt.plot(fake_data.numpy()[:, 0], fake_prediction.data.numpy(), 'b-')
        plt.text(0, 3, 'Train loss=%.4f' % train_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.text(0, 2.5, 'Test loss=%.4f' % test_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    # use least square to determine C1 and C2 based on neural network output
    epoch += 1
    C1_C2 = curve_fit(I_P, fake_prediction.data.numpy().astype(np.float64)[:155], fake_data.numpy().astype(np.float64)[:155, 0], bounds=(0, np.inf))[0]
    # convert to float64, or curve_fit doesn't work
    print(C1_C2)
    a = []
    b = []
    for ii in range(1, 33):
        if ii == 1:
            a.append(list(0.2 / I_P(0.2, C1_C2[0], C1_C2[1]) / 225 * np.ones((225,))))
            b.append(0)
        else:
            a.append(list(0.1 / (I_P((ii + 1) / 10, C1_C2[0], C1_C2[1]) - I_P(ii / 10, C1_C2[0], C1_C2[1])) / 225 * np.ones((225,))))
            b.append(ii / 10 - 0.1 / (I_P((ii + 1) / 10, C1_C2[0], C1_C2[1]) - I_P(ii / 10, C1_C2[0], C1_C2[1])) * I_P(ii / 10, C1_C2[0], C1_C2[1]))
    # use determined physical principle to initialize neural network for next iteration
    net.hidden.weight.data = torch.Tensor(a)
    net.hidden.bias.data = torch.Tensor(b)

plt.ioff()
plt.show()
print(epoch)                # converge in 961 steps, train loss=0.0110, test loss=0.0111, C1=105.4, C2=17.24
