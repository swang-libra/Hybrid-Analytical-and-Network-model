import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

torch.manual_seed(7)    # reproducible

image_patches_for_training = torch.from_numpy(np.load('balance sensor experimental pixel values for training.npy')).type(torch.FloatTensor)
pressure_values_for_training = torch.from_numpy(np.load('balance sensor experimental pressure values for training.npy')).type(torch.FloatTensor)
image_patches_for_test = torch.from_numpy(np.load('balance sensor experimental pixel values for test.npy')).type(torch.FloatTensor)
pressure_values_for_test = torch.from_numpy(np.load('balance sensor experimental pressure values for test.npy')).type(torch.FloatTensor)
ideal_image_patches = torch.from_numpy(np.load('balance sensor ideal pixel values for calculating pressure-pixel relationship.npy')).type(torch.FloatTensor)
pressure_values_for_ideal_image_patches = None


# this is the qualitative physical model between pressure and pixel value
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
    print(iteration, epoch)            # during 1st iteration, neural network parameters are set based on default random initialization
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate[iteration])
    train_loss_before = 1e5
    train_loss = train_loss_before - 1
    while abs(train_loss_before - train_loss) > tolerance[iteration]:
        epoch += 1
        train_loss_before = train_loss
        train_prediction = net(image_patches_for_training)
        train_loss = loss_func(train_prediction, pressure_values_for_training)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        plt.cla()
        plt.plot(image_patches_for_training.numpy().min(axis=1), pressure_values_for_training.numpy(), 'y.')
        plt.plot(image_patches_for_training.numpy().max(axis=1), pressure_values_for_training.numpy(), 'k.')
        plt.xlabel('pixel value')
        plt.ylabel('pressure/kg/cm2')
        plt.grid(True)
        test_prediction = net(image_patches_for_test)
        test_loss = loss_func(test_prediction, pressure_values_for_test)
        pressure_values_for_ideal_image_patches = net(ideal_image_patches)
        plt.plot(ideal_image_patches.numpy()[:, 0], pressure_values_for_ideal_image_patches.data.numpy(), 'b-')
        plt.text(0, 3, 'Train loss=%.4f' % train_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.text(0, 2.5, 'Test loss=%.4f' % test_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    # use least square to determine C1 and C2 based on neural network output
    epoch += 1
    # convert to float64, or curve_fit doesn't work
    # here the slice ([:155]) is for consistency with experiment data because the maximum pixel value in experiment data is 154
    C1_C2 = curve_fit(I_P, pressure_values_for_ideal_image_patches.data.numpy().astype(np.float64)[:155], ideal_image_patches.numpy().astype(np.float64)[:155, 0], bounds=(0, np.inf))[0]
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
    # use quantitative physical model to initialize neural network for next iteration
    net.hidden.weight.data = torch.Tensor(a)
    net.hidden.bias.data = torch.Tensor(b)

plt.ioff()
plt.show()
print(epoch)                # converge in 961 steps, train loss=0.0110, test loss=0.0111, C1=105.4, C2=17.24
