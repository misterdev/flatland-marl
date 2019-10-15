import torch.nn as nn
import torch.nn.functional as F


# Number of Linear input connections depends on output of conv2d layers
# and therefore the input image size, so compute it.
def conv2d_size_out(in_size, kernel_size, stride):
    return (in_size - kernel_size) // stride + 1

# See 'Playing Atari with Deep Reinforcement Learning"
# https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# choose as width and height env max width and env max height (as specified in the challenge)
class ConvQNetwork(nn.Module):
    def __init__(self, width=200, height=200, action_size=5, hidsize1=128, hidsize2=64):
        super(ConvQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(22, 16, kernel_size=3)  # input_channels, output_channels, kernel_size, stride, padding
        # applies batch normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
        # param is num_features
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(in_size, kernel_size=5, stride=3):
            return (in_size - kernel_size) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 3, 1)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 3, 1)))
        linear_input_size = conv_width * conv_height

        self.val1 = nn.Linear(linear_input_size, hidsize1)  # in_features, out_features, i valori sono precalcolati, se sono giusti
        self.val2 = nn.Linear(hidsize1, hidsize2)
        self.val3 = nn.Linear(hidsize2, 1)

        self.adv1 = nn.Linear(linear_input_size, hidsize1)
        self.adv2 = nn.Linear(hidsize1, hidsize2)
        self.adv3 = nn.Linear(hidsize2, action_size)


    # Return q-values for actions
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x - (1, 64, 21, 21), 21*21 = 441
        # Value function approximation
        val = F.relu(self.val1(x.view(64, -1)))
        val = F.relu(self.val2(val))
        val = self.val3(val)

        # Advantage calculation
        adv = F.relu(self.adv1(x.view(64, -1)))
        adv = F.relu(self.adv2(adv))
        adv = self.adv3(adv)
        # (64, 1) + (64, 5) + boh
        return val + adv - adv.mean()


# implementazione basata sul paper originale https://github.com/dxyang/DQN_pytorch
class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=4) # TODO kernel_size at this layer should be 8
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        conv_width = conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_channels, 4, 4), 4, 2), 3, 1)
        
        #self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        #self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.fc1_adv = nn.Linear(in_features=conv_width * conv_height * 64, out_features=512) # TODO 64 sono gli out_channels??
        self.fc1_val = nn.Linear(in_features=conv_width * conv_height * 64, out_features=512)
        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x
    