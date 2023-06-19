import torch
from torch import nn
from torch.nn import functional as F

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma
####################3


class REncoder(torch.nn.Module):
    """Encodes inputs of the form (x_i,y_i) into representations, r_i."""

    def __init__(self, in_dim, h_dim, out_dim, init_func=torch.nn.init.normal_):
        super(REncoder, self).__init__()
        self.l1_size = h_dim

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, out_dim)

        self.a = torch.nn.ReLU()

        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)

    def forward(self, inputs):
        return self.l2(self.a(self.l1(inputs)))


class ZEncoder(torch.nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the
    normally distributed function encoding, z."""

    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(ZEncoder, self).__init__()
        self.m1_size = out_dim
        self.std1_size = out_dim

        self.m1 = torch.nn.Linear(in_dim, self.m1_size)
        self.std1 = torch.nn.Linear(in_dim, self.m1_size)

        if init_func is not None:
            init_func(self.m1.weight)
            init_func(self.std1.weight)

    def forward(self, inputs):
        softplus = torch.nn.Softplus()
        return self.m1(inputs), softplus(self.std1(inputs))


class Decoder_(torch.nn.Module):
    """
    Takes the x star points, along with a 'function encoding', z, and makes predictions.
    """

    def __init__(self, in_dim, g_dim1, g_dim2, out_dim, init_func=torch.nn.init.normal_):
        super(Decoder_, self).__init__()
        self.l1_size = g_dim1
        self.l2_size = g_dim2

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, out_dim)

        self.m1 = torch.nn.Linear(out_dim, out_dim)
        self.std1 = torch.nn.Linear(out_dim, out_dim)


        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)

        # self.a = torch.nn.Sigmoid()

    def forward(self, x_pred, z):
        """x_pred: No. of data points, by x_dim
        z: No. of samples, by z_dim
        """

        # zs_reshaped = z.expand(z.shape[0], z.shape[1], x_pred.shape[0])

        # zs_reshaped = z.unsqueeze(0).repeat(x_pred.shape[0], 1, 1)
        zs_reshaped = z.unsqueeze(1).repeat(1, x_pred.shape[1], 1)





        xz = torch.cat([x_pred, zs_reshaped], dim=2)
        softplus = torch.nn.Softplus()
        mu = self.m1(self.l2(self.l1(xz)))

        sigma = .0001 * softplus(self.std1(self.l2(self.l1(xz))))

        return mu , sigma * .01