import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

torch.manual_seed(7)    # reproducible

experiment_data = scio.loadmat('skin sensor experiment data.mat')
circle_size = experiment_data['circle_size'] * 1e-6
max_FOV = circle_size.max()
min_FOV = circle_size.min()
circle_size = torch.from_numpy(circle_size).type(torch.FloatTensor)

# the weight of skin sensor lens and cap is 2.65g, the unit of pressure is N
force = (np.squeeze(experiment_data['pressure']) + 2.65) * 0.0098
max_force = force.max()
min_force = force.min()
force = torch.from_numpy(force).type(torch.FloatTensor)

fake_pixel_num = torch.from_numpy(np.arange(60000, 85000, 1000).repeat(5).reshape(-1, 5)).type(torch.FloatTensor) * 1e-6


def P_N(N, C1, C2):
    return C2 - C1 / np.sqrt(N)


class Regression_net2(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Regression_net2, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.predict(x)[:, 0]


net = Regression_net2(5, 32, 1)
net2_params_from_bs = torch.load('Type 2 net_params transferred from balance sensor network.pkl')
max_pixel = 154
min_pixel = 0
max_pressure = 3.84
min_pressure = 0
net.hidden.weight.data = (net2_params_from_bs['hidden.weight'] * (max_pixel - min_pixel) / (max_FOV - min_FOV)).mean(dim=1).unsqueeze(dim=1).repeat(1, 5) * 225 / 5
net.hidden.bias.data = (net2_params_from_bs['hidden.bias'] - (max_pixel * min_FOV - min_pixel * max_FOV) / (max_FOV - min_FOV) * net2_params_from_bs['hidden.weight'].sum(dim=1))
net.predict.weight.data = net2_params_from_bs['predict.weight'] * (max_force - min_force) / (max_pressure - min_pressure)
net.predict.bias.data = (net2_params_from_bs['predict.bias'] - min_pressure) * (max_force - min_force) / (max_pressure - min_pressure) + min_force

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
tolerance = [1e-6, 1e-7]

epoch = 0
for iteration in range(8):
    print(iteration, epoch)
    train_loss_before = 10
    train_loss = train_loss_before - 1
    while abs(train_loss_before - train_loss) > tolerance[min(iteration, 1)]:
        epoch += 1
        train_loss_before = train_loss
        train_prediction = net(circle_size)
        train_loss = loss_func(train_prediction, force)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        plt.cla()
        plt.plot(circle_size.numpy().ravel('C') * 1e6, force.numpy().repeat(5), '.k')
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
    weight1 = []
    bias1 = []
    weight2 = [[]]
    bias2 = []
    a = []
    for ii in range(0, 32):
        N1 = fake_pixel_num.numpy()[0, 0] * (12 - ii) / 12 + fake_pixel_num.numpy()[-1, 0] * ii / 12
        P1 = P_N(N1, C1_C2[0], C1_C2[1])
        N2 = fake_pixel_num.numpy()[0, 0] * (11 - ii) / 12 + fake_pixel_num.numpy()[-1, 0] * (ii + 1) / 12
        P2 = P_N(N2, C1_C2[0], C1_C2[1])
        a.append((P2 - P1) / (N2 - N1))
        if ii == 0:
            weight1.append(list(abs(a[0]) / 5 * np.ones((5,))))
            bias1.append(0)
            weight2[0].append(a[0] / abs(a[0]))
            bias2.append(P1 - (P2 - P1) / (N2 - N1) * N1)
        else:
            weight1.append(list(abs(a[ii]-a[ii-1]) / 5 * np.ones((5,))))
            bias1.append(-abs(a[ii]-a[ii-1]) * N1)
            weight2[0].append((a[ii] - a[ii - 1]) / abs(a[ii] - a[ii - 1]))

    net.hidden.weight.data = torch.Tensor(weight1)
    net.hidden.bias.data = torch.Tensor(bias1)
    net.predict.weight.data = torch.Tensor(weight2)
    net.predict.bias.data = torch.Tensor(bias2)

plt.show()
print(epoch)
