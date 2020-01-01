import torch.nn as nn
import torch.nn.functional as F

# TODO Must return actions for ALL agents ?

'''
Functions to compute output width and height (considering they could not 'square') after convolution.
'dim' can be 'width' or 'height'
'''
def dim_output(input_dim, filter_dim, stride_dim):
    return (input_dim - filter_dim) // stride_dim + 1

class Dueling_DQN(nn.Module):
    def __init__(self, width, height, action_space):
        super(Dueling_DQN, self).__init__()
        self.action_space = action_space
        # input shape (batch_size, in_channels = height/num_rails, width/prediction_depth + 1) 
        self.conv1 = nn.Conv1d(in_channels=height, out_channels=64, kernel_size=20)
        # output shape (batch_size, out_channels, conv_width)
        conv_width = dim_output(input_dim=width, filter_dim=20, stride_dim=1)

        # in_features = conv_width * out_channels (feature maps/number of kernels, arbitrary)
        # after last Conv1d
        self.fc1_adv = nn.Linear(in_features=conv_width * 64, out_features=512) 
        self.fc1_val = nn.Linear(in_features=conv_width * 64, out_features=512)
        self.fc2_adv = nn.Linear(in_features=512, out_features=action_space)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x): # 
        # batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.action_space)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_space)
        return x
