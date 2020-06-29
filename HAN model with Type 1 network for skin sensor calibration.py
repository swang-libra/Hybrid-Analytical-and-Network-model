import numpy as np
import torch
import torch.nn as nn
import scipy.io as scio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

torch.manual_seed(7)    # reproducible

experiment_data = scio.loadmat('skin sensor experiment data.mat')
circle_size = torch.from_numpy(experiment_data['circle_size'].astype(np.float32)) * 1e-6
# the weight of skin sensor lens and cap is 2.65g, the unit of pressure is N
pressure = torch.from_numpy((np.squeeze(experiment_data['pressure']).astype(np.float32) + 2.65) * 0.0098)
fake_pixel_num = torch.from_numpy(np.arange(60000, 85000, 1000).repeat(5).reshape(-1, 5)).type(torch.FloatTensor) * 1e-6


def P_N(N, C1, C2):
    return C2 - C1 / np.sqrt(N)


class Regression_net1(nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Regression_net1, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)

    def forward(self, x):
        return torch.min(self.hidden(x), dim=1)[0]


net = Regression_net1(5, 32)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
tolerance = [1e-6, 1e-7]

epoch = 0
for iteration in range(6):
    print(iteration, epoch)
    train_loss_before = 10
    train_loss = train_loss_before - 1
    while abs(train_loss_before - train_loss) > tolerance[min(iteration, 1)]:
        epoch += 1
        train_loss_before = train_loss
        train_prediction = net(circle_size)
        train_loss = loss_func(train_prediction, pressure)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        plt.cla()
        plt.plot(circle_size.numpy().ravel('C') * 1e6, pressure.numpy().repeat(5), '.k')
        plt.xlabel('pixel number inside FOV')
        plt.ylabel('contact force/N')
        plt.xlim((6e4, 8.5e4))
        plt.grid(True)
        fake_pressure = net(fake_pixel_num)
        plt.plot(fake_pixel_num.numpy()[:, 0] * 1e6, fake_pressure.data.numpy(), 'b-')
        plt.text(70000, 0.04, 'Train loss=%.6f' % train_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

    epoch += 1
    C1_C2 = curve_fit(P_N, fake_pixel_num.numpy().astype(np.float64)[:, 0], fake_pressure.data.numpy().astype(np.float64), bounds=(0, np.inf))[0]
    # convert to float64, or curve_fit doesn't work
    print(C1_C2, train_loss.data.numpy())
    a = []
    b = []
    for ii in range(0, 32):
        N1 = fake_pixel_num.numpy()[0, 0] * (12 - ii) / 12 + fake_pixel_num.numpy()[-1, 0] * ii / 12
        P1 = P_N(N1, C1_C2[0], C1_C2[1])
        N2 = fake_pixel_num.numpy()[0, 0] * (11 - ii) / 12 + fake_pixel_num.numpy()[-1, 0] * (ii + 1) / 12
        P2 = P_N(N2, C1_C2[0], C1_C2[1])
        a.append(list((P2 - P1) / (N2 - N1) / 5 * np.ones((5,))))
        b.append(P1 - (P2 - P1) / (N2 - N1) * N1)

    net.hidden.weight.data = torch.Tensor(a)
    net.hidden.bias.data = torch.Tensor(b)

plt.show()
print(epoch)
