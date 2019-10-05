import torch.nn as nn
import torch.nn.functional as F


class ConvQNetwork(nn.Module):
    def __init__(self, height, width, outputs, action_size=5, hidsize1=128, hidsize2=128):
        super(ConvQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(22, 16, kernel_size=5, stride=2)  # input_channels, output_channels, kernel_size, stride
        # applies batch normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
        # param is num_features
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.val1 = nn.Linear(6400, hidsize1) # in_features, out_features
        self.val2 = nn.Linear(hidsize1, hidsize2)
        self.val3 = nn.Linear(hidsize2, 1)

        self.adv1 = nn.Linear(6400, hidsize1)
        self.adv2 = nn.Linear(hidsize1, hidsize2)
        self.adv3 = nn.Linear(hidsize2, action_size)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_width * conv_height * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Return q-values for actions
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Value function approximation
        val = F.relu(self.val1(x.view(x.size(0), -1)))
        val = F.relu(self.val2(val))
        val = self.val3(val)

        # Advantage calculation
        adv = F.relu(self.adv1(x.view(x.size(0), -1)))
        adv = F.relu(self.adv2(adv))
        adv = self.adv3(adv)
        return val + adv - adv.mean()

