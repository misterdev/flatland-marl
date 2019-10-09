import copy
import os
import random
from collections import namedtuple, deque, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_training.model import Dueling_DQN

# Params for ReplayBuffer class
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32  # minibatch size = 512

GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4
UPDATE_EVERY = 10  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, double_dqn=True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        #self.state_size = state_size # used by the network, not the algorithm
        #self.env_width = env_width
        #self.env_height = env_height
        self.action_size = action_size
        # in double_dqn there are local and target networks, 
        self.double_dqn = double_dqn
        # Q-Network
        #self.qnetwork_local = ConvQNetwork().to(device) 
        self.qnetwork_local = Dueling_DQN(in_channels=22, num_actions=5).to(device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        # used for training
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))

    def step(self, state, action, reward, next_state, done, train=True):
        # Save experience in replay memory
        # state: (22, 200, 200) tutta la griglia + 22 channels
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        # training is done taking a sample batch from the replay memory
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                if train:
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # This add a first dimension to state? why? TODO
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        # action_values: (64, 5) should be (1,5)!!
        with torch.no_grad(): 
            action_values = self.qnetwork_local(state) # retrieve q action_values as computed by the NN
        self.qnetwork_local.train() # used by pytorch to set the module in training mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # select action with max value
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            Q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

            # Compute Q targets for current states

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    # This same function is used for states, actions, rewards etc, so the parameter 'states' doesn't contain states alll the time
    # and for this reason has different shapes
    def __v_stack_impr(self, states):
        #sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        # means states are actually states (not actions, or rewards...)
        if isinstance(states[0], Iterable):
            sub_dim = len(states[0][0])
            np_states = np.reshape(np.array(states), (len(states), sub_dim, 200, 200))
        else:
            sub_dim = 1
            np_states = np.reshape(np.array(states), (len(states), sub_dim))

        # np.array create a 1d array of states and reshape tries to reshape it into (512, 22)
        # states qua è una lista che contiene 512 array di shape (1, 22, 200, 200)
        # quando viene linearizzato con np.array quindi diventa 512 * 22 * 200 * 200
        # questo in pratica serve per ritornate l'exp in batch di batch_size specificata
        # np_states = np.reshape(np.array(states), (len(states), sub_dim, 200, 200))
        return np_states
