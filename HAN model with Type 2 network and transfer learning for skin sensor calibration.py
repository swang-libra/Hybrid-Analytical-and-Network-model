import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

torch.manual_seed(7)    # reproducible

# narrow down the dimension of FOV_size values to match the dimension of contact_force for convenience of training network
FOV_size = np.load('skin sensor experimental FOV size values.npy') * 1e-6
max_FOV_size = FOV_size.max()
min_FOV_size = FOV_size.min()
FOV_size = torch.from_numpy(FOV_size).type(torch.FloatTensor)

# the weight of skin sensor glass plate is 2.65g, the unit of contact_force is N
contact_force = (np.load('skin sensor experimental contact force values.npy') + 2.65) * 0.0098
max_contact_force = contact_force.max()
min_contact_force = contact_force.min()
contact_force = torch.from_numpy(contact_force).type(torch.FloatTensor)

# input man-made ideal FOV size values into network to calculate the force-FOV relationship
ideal_FOV_size = torch.from_numpy(np.arange(60000, 85000, 1000).repeat(5).reshape(-1, 5)).type(torch.FloatTensor) * 1e-6
contact_force_for_ideal_FOV_size = None


# this is the qualitative physical model between contact force and FOV size
def F_N(N, C1, C2):
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
# balance sensor experiment data range
max_pixel_value = 154
min_pixel_value = 0
max_pressure_value = 3.84
min_pressure_value = 0
# use transfer learning to initialize network
net.hidden.weight.data = (net2_params_from_bs['hidden.weight'] * (max_pixel_value - min_pixel_value) / (max_FOV_size - min_FOV_size)).mean(dim=1).unsqueeze(dim=1).repeat(1, 5) * 225 / 5
net.hidden.bias.data = (net2_params_from_bs['hidden.bias'] - (max_pixel_value * min_FOV_size - min_pixel_value * max_FOV_size) / (max_FOV_size - min_FOV_size) * net2_params_from_bs['hidden.weight'].sum(dim=1))
net.predict.weight.data = net2_params_from_bs['predict.weight'] * (max_contact_force - min_contact_force) / (max_pressure_value - min_pressure_value)
net.predict.bias.data = (net2_params_from_bs['predict.bias'] - min_pressure_value) * (max_contact_force - min_contact_force) / (max_pressure_value - min_pressure_value) + min_contact_force

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
        train_prediction = net(FOV_size)
        train_loss = loss_func(train_prediction, contact_force)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        plt.cla()
        plt.plot(FOV_size.numpy().ravel('C') * 1e6, contact_force.numpy().repeat(5), '.k')
        plt.xlabel('pixel number inside FOV')
        plt.ylabel('contact force/N')
        plt.xlim((6e4, 8.5e4))
        plt.grid(True)
        contact_force_for_ideal_FOV_size = net(ideal_FOV_size)
        plt.plot(ideal_FOV_size.numpy()[:, 0] * 1e6, contact_force_for_ideal_FOV_size.data.numpy(), 'b-')
        plt.text(70000, 0.04, 'Train loss=%.6f' % train_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    # use least square to determine C1 and C2 based on neural network output
    epoch += 1
    # convert to float64, or curve_fit doesn't work
    C1_C2 = curve_fit(F_N, ideal_FOV_size.numpy().astype(np.float64)[:, 0], contact_force_for_ideal_FOV_size.data.numpy().astype(np.float64), bounds=(0, np.inf))[0]
    print(C1_C2, train_loss.data.numpy())
    weight1 = []
    bias1 = []
    weight2 = [[]]
    bias2 = []
    a = []
    for ii in range(0, 32):
        N1 = ideal_FOV_size.numpy()[0, 0] * (12 - ii) / 12 + ideal_FOV_size.numpy()[-1, 0] * ii / 12
        F1 = F_N(N1, C1_C2[0], C1_C2[1])
        N2 = ideal_FOV_size.numpy()[0, 0] * (11 - ii) / 12 + ideal_FOV_size.numpy()[-1, 0] * (ii + 1) / 12
        F2 = F_N(N2, C1_C2[0], C1_C2[1])
        a.append((F2 - F1) / (N2 - N1))
        if ii == 0:
            weight1.append(list(abs(a[0]) / 5 * np.ones((5,))))
            bias1.append(0)
            weight2[0].append(a[0] / abs(a[0]))
            bias2.append(F1 - (F2 - F1) / (N2 - N1) * N1)
        else:
            weight1.append(list(abs(a[ii]-a[ii-1]) / 5 * np.ones((5,))))
            bias1.append(-abs(a[ii]-a[ii-1]) * N1)
            weight2[0].append((a[ii] - a[ii - 1]) / abs(a[ii] - a[ii - 1]))
    # use quantitative physical model to initialize neural network for next iteration
    net.hidden.weight.data = torch.Tensor(weight1)
    net.hidden.bias.data = torch.Tensor(bias1)
    net.predict.weight.data = torch.Tensor(weight2)
    net.predict.bias.data = torch.Tensor(bias2)

plt.show()
print(epoch)
