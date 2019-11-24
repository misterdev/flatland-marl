import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidsize1=256, hidsize2=128):
        super(QNetwork, self).__init__()

        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc3_val = nn.Linear(hidsize2, 1)

        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc3_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc3_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc3_adv(adv)
        return val + adv - adv.mean()


'''
Functions to compute output width and height (considering they could not 'square') after convolution.
'dim' can be 'width' or 'height'
'''
def dim_output(input_dim, filter_dim, stride_dim):
    return (input_dim - filter_dim) // stride_dim + 1

# es width = 15, height = 30
# Dueling DQN - implementation based on original paper https://github.com/dxyang/DQN_pytorch
class ConvQNetwork(nn.Module):
    def __init__(self, in_channels, action_space):
        super(ConvQNetwork, self).__init__()
        self.action_space = action_space

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2)  # I changed the kernel sizes of each layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)

        conv_width = dim_output(dim_output(dim_output(15, 4, 2), 2, 2), 1, 1)
        conv_height = dim_output(dim_output(dim_output(30, 4, 2), 2, 2), 1, 1)

        # self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        # self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        # in_features = conv_width * conv_height * out_channels (feature maps) of the last conv layer
        self.fc1_adv = nn.Linear(in_features=conv_width * conv_height * 64, out_features=128) 
        self.fc1_val = nn.Linear(in_features=conv_width * conv_height * 64, out_features=128)
        self.fc2_adv = nn.Linear(in_features=128, out_features=action_space)
        self.fc2_val = nn.Linear(in_features=128, out_features=1)

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
        val = self.fc2_val(val).expand(x.size(0), self.action_space)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_space)
        return x
