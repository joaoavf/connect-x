from collections import deque

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from random import sample, random
import wandb
import numpy as np
from tqdm import tqdm
import time
from kaggle_environments import evaluate, make
import ipdb


@dataclass
class SARSD:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class DQNAgent:
    def __init__(self, model):
        self.model = model

    def get_actions(self, observations):
        q_vals = self.model(observations)

        return q_vals.max(-1)[1]


class ReplayBuffer:
    def __init__(self, buffer_size=1_000_000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def insert(self, sars):
        self.buffer.append(sars)

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)


class Model(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(Model, self).__init__()
        assert len(obs_shape) == 1, "This network only works for flat obs"

        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions),
            torch.nn.Tanh()
        )

        for layer in self.net[2:-2:2]:
            torch.nn.init.kaiming_normal_(layer.weight)

        torch.nn.init.normal_(self.net[-2].weight)

        self.optim = optim.AdamW(self.net.parameters(), lr=0.001)

    def forward(self, x):
        return self.net(x)

    def get_action(self, observation, epsilon, device):
        prediction = self(torch.Tensor([observation])).to(device)[0].detach().numpy()  # .max(-1)[-1].item()

        if np.random.random() < epsilon:
            return int(
                np.random.choice([c for i, c in enumerate(range(self.num_actions)) if observation[i] == 0])), np.max(
                prediction)

        else:
            for i in range(self.num_actions):
                if observation[i] != 0:
                    prediction[i] = -1
            return int(np.argmax(prediction)), np.max(prediction)


def train_step(model, target, state_transitions, num_actions, device, discount_factor=0.99):
    cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions]).to(device)
    rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions]).to(device)
    mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]).to(device)
    next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions]).to(device)
    actions = [s.action for s in state_transitions]

    move_validity = next_states[:, :num_actions] == 0
    with torch.no_grad():
        q_values_next = target(next_states)
    q_values_next = -np.where(move_validity, q_values_next, -1).max(-1)

    model.optim.zero_grad()
    qvals = model(cur_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    actual_values = rewards[:, 0] + mask[:, 0] * q_values_next * discount_factor

    expected_values = torch.sum(qvals * one_hot_actions, -1)

    loss = ((actual_values - expected_values) ** 2).mean()

    loss.backward()
    model.optim.step()

    return loss


def update_target_model(model, target):
    target.load_state_dict(model.state_dict())


def preprocess(observation):
    board = observation['board']
    player = observation['mark']
    return np.array([1 if value == player else 0 if value == 0 else -1 for value in board])


class Game:
    def __init__(self, test=False, checkpoint=None, device='cpu'):
        if not test:
            wandb.init(project="dqn-tutorial", name="dqn-minimax")

        self.tq = tqdm()

        self.min_rb_size = 1_000
        self.sample_size = 512

        self.test = test
        self.checkpoint = checkpoint
        self.device = device

        self.eps_min = 0.1
        self.eps_decay = 0.999999

        self.env_steps_before_train = 64
        self.tgt_model_update = 250

        self.env = make('connectx', debug=False)
        self.configuration = self.env.configuration
        self.action_space = gym.spaces.Discrete(self.configuration.columns)
        self.observation_space = np.array([0] * self.configuration.columns * self.configuration.rows)

        self.last_observation = self.env.reset()[0]['observation']

        self.last_observation = preprocess(observation=self.last_observation)

        self.model = Model(self.observation_space.shape, self.action_space.n).to(device)
        self.target = Model(self.observation_space.shape, self.action_space.n).to(device)

        if checkpoint is not None:
            print('Models loaded:')
            self.model.load_state_dict(torch.load(checkpoint))
            self.target.load_state_dict(torch.load(checkpoint))

        self.rb = ReplayBuffer()
        self.steps_since_train = 0
        self.epochs_since_tgt = 0

        self.step_num = -1 * self.min_rb_size

        self.episode_rewards = []

        self.rolling_reward = []
        self.active_player = 0

    def play(self):

        if self.test:
            self.env.render()
            time.sleep(0.01)

        self.tq.update(1)
        eps = max(self.eps_decay ** self.step_num, self.eps_min)
        if self.test:
            eps = 0

        action, prediction = self.model.get_action(observation=self.last_observation, epsilon=eps, device=self.device)

        p_dict = self.env.step([action if i == self.active_player else None for i in [0, 1]])

        reward = p_dict[self.active_player]['reward']
        done = self.env.done

        observation = p_dict[[1, 0][self.active_player]]['observation']
        observation = preprocess(observation=observation)

        if done:
            if reward == 1:  # Won
                reward = 1
            elif reward == 0:  # Lost
                reward = -1
            else:  # Draw
                reward = 0
        else:
            reward = 0

        self.active_player = [1, 0][self.active_player]

        self.rolling_reward.append(prediction)

        self.rb.insert(SARSD(self.last_observation, action, reward, observation, done))

        if done:
            self.episode_rewards.append(np.mean(self.rolling_reward))
            if self.test:
                print(self.rolling_reward)
            self.rolling_reward = []
            observation = self.env.reset()[0]['observation']
            observation = preprocess(observation=observation)

            self.active_player = 0

        self.last_observation = observation
        self.steps_since_train += 1
        self.step_num += 1

        if not self.test and self.step_num > self.min_rb_size and self.steps_since_train > self.env_steps_before_train:
            loss = train_step(self.model, self.target, self.rb.sample(self.sample_size), self.action_space.n,
                              self.device)
            wandb.log({'loss': loss.detach().cpu().item(),
                       'eps': eps,
                       'rewards': np.mean(self.episode_rewards)
                       },
                      step=self.step_num)

            self.episode_rewards = []
            self.epochs_since_tgt += 1

            if self.epochs_since_tgt > self.tgt_model_update:
                update_target_model(self.model, self.target)
                torch.save(self.target.state_dict(), f'models/dqn_minimax_{self.step_num}.pth')
                self.epochs_since_tgt = 0

            self.steps_since_train = 0


def main(test=False, checkpoint=None, device='cpu'):
    game = Game(test=test, checkpoint=checkpoint, device=device)
    try:
        while True:
            game.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()  # (checkpoint='models/dqn_minimax_1770935.pth')
