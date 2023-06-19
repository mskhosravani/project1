import torch
from random import randint
from neural_process import NeuralProcessImg
from torch import nn
from torch.distributions.kl import kl_divergence
from utils import (context_target_split, batch_context_target_mask,
                   img_mask_to_np_input)

mse = nn.MSELoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")


class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = isinstance(self.neural_process, NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []
    def train(self, data_loader, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.

            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()


                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    (mag, phas, img), _ = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img #torch.concatenate((mag, phas), dim=1)
                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)
                    print("the range of input for np", torch.max(img))

                    p_y_pred, q_target, q_context = \
                        self.neural_process(img, context_mask, target_mask)
                    print("the range of input for np prediction", torch.max(p_y_pred.loc))

                    # Calculate y_target as this will be required for loss
                    _, y_target = img_mask_to_np_input(img, target_mask)
                    print("the range of input for np target", torch.max(y_target))
                else:
                    x, y = data
                    x_context, y_context, x_target, y_target = \
                        context_target_split(x, y, num_context, num_extra_target)
                    p_y_pred, q_target, q_context = \
                        self.neural_process(x_context, y_context, x_target, y_target)

                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()


                epoch_loss += loss.item()


                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl
#############temporal
def reshape_images(img):
    """ Reshape images from (b_size, ch, dim, dim) to (b_size, dim*dim, [ind, y1, y2, y3]) """
    b_size, ch, dim, _ = img.shape

    # Transpose the tensor so the channel dimension comes last, and then reshape it
    img = img.transpose(1, 3).reshape(b_size, dim * dim, ch)

    # Generate a tensor with the indices of each pixel
    x = torch.arange(dim).float() / (dim - 1)
    y = torch.arange(dim).float() / (dim - 1)
    grid = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1, 2)
    grid = grid.repeat(b_size, 1, 1)

    # Concatenate the rescaled indices with the pixel values
    img = torch.cat((grid, img), dim=2)

    return img
def reconstruct_images(reshaped_imgs, dim):
    """ Reshape images from (b_size, dim*dim, [ind, y1, y2, y3]) to (b_size, ch, dim, dim) """
    b_size = reshaped_imgs.shape[0]

    # Separate the pixel values from the indices
    img = reshaped_imgs[:, :, 2:]
    print(img.shape)

    # Resize and transpose the images back to original shape
    img = img.reshape(b_size, dim, dim, -1).transpose(1, 3)

    return img
def select_pixels(img_reshaped, num_pixels):
    """ Randomly select a portion of the pixels from each reshaped image. """
    b_size, dd, _ = img_reshaped.shape

    # Generate a random permutation of the pixel indices
    perm = torch.randperm(dd)

    # Select the desired number of pixels
    selected_pixels = img_reshaped[:, perm[:num_pixels], :]

    # Get the remaining pixels
    remaining_pixels = img_reshaped[:, perm[num_pixels:], :]

    return selected_pixels, remaining_pixels

###############################3
class NPTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = isinstance(self.neural_process, NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []
    def train(self, data_loader, n_epochs):
        losses = []
        for t in range(n_epochs):

            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                (mag, phas, img), _ = data  # data is a tuple (img, label)


                img_reshaped = reshape_images(phas)  # Reshape the images
                x_target, y_target = img_reshaped[:,:,:2], img_reshaped[:,:,2:]



                # img_reconstructed = reconstruct_images(img_reshaped,
                #                                        32)  # Reconstruct the original images
                num_pixels = torch.randint(100, x_target.shape[1]-100, size=(1,)).item()
                selected_pixels, _ = select_pixels(img_reshaped, num_pixels)
                x_context, y_context = selected_pixels[:,:,:2], selected_pixels[:,:,2:]












                mu, std, z_mean_all, z_std_all, z_mean_context, z_std_context = \
                    self.neural_process(x_context.to(self.device), y_context.to(self.device),
                                        x_target.to(self.device), y_target.to(self.device))
                print(torch.max(mu))
                # print(y_target.shape)
                # print(z_mean_all.shape)
                # print(z_mean_context.shape)



                # Compute loss and backprop


                loss = self.log_likelihood(mu, std, y_target.to(self.device)) + self.kl_divergence(z_mean_all, z_std_all,
                                                                    z_mean_context, z_std_context)
                # print(loss)

                losses.append(loss)

                loss.backward()

                self.optimizer.step()


    def log_likelihood(self, mu, std, target):
        l = mse(target, mu)


        return l

        # norm = torch.distributions.normal.Normal(mu, std)
        #
        # return norm.log_prob(target).sum(dim=0).mean()


    def KLD_gaussian(self, mu_q, std_q, mu_p, std_p):
        """Analytical KLD between 2 Gaussians."""
        qs2 = std_q ** 2 + 1e-16
        ps2 = std_p ** 2 + 1e-16
        kl = (qs2 / ps2 + ((mu_q - mu_p) ** 2) / ps2 + torch.log(ps2 / qs2) - 1.0).sum() * 0.5
        # print(kl)

        return kl

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        """Calculate KL divergence between two normal distributions."""
        kl_div = kl_loss(mu1, mu2)


        return kl_div.mean()
    def sigma(self, sig):
        return sig.mean()


