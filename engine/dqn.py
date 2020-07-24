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
    def __init__(self, buffer_size=100_000):
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
            torch.nn.Linear(obs_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions)
        )

        self.optim = optim.Adam(self.net.parameters(), lr=0.01)

    def forward(self, x):
        return self.net(x)


def train_step(model, target, state_transitions, num_actions, device, discount_factor=0.99):
    cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions]).to(device)
    rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions]).to(device)
    mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]).to(device)
    next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions]).to(device)
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        q_vals_next = target(next_states).max(-1)[0]

    model.optim.zero_grad()
    qvals = model(cur_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss = ((rewards + mask[:, 0] * q_vals_next * discount_factor - torch.sum(qvals * one_hot_actions, -1)) ** 2).mean()
    loss.backward()
    model.optim.step()

    return loss


def update_target_model(model, target):
    target.load_state_dict(model.state_dict())


class Game:
    def __init__(self, test=False, checkpoint=None, device='cpu'):
        if not test:
            wandb.init(project="dqn-tutorial", name="dqn-cartpole")

        self.tq = tqdm()

        self.min_rb_size = 10000
        self.sample_size = 2500

        self.test = test
        self.checkpoint = checkpoint
        self.device = device

        self.eps_min = 0.01
        self.eps_decay = 0.999999

        self.env_steps_before_train = 100
        self.tgt_model_update = 500

        self.env = gym.make('CartPole-v1')
        self.last_observation = self.env.reset()

        self.model = Model(self.env.observation_space.shape, self.env.action_space.n).to(device)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
        self.target = Model(self.env.observation_space.shape, self.env.action_space.n).to(device)

        self.rb = ReplayBuffer()
        self.steps_since_train = 0
        self.epochs_since_tgt = 0

        self.step_num = -1 * self.min_rb_size

        self.episode_rewards = []
        self.rolling_reward = 0

    def play(self):
        if self.test:
            self.env.render()
            time.sleep(0.01)

        self.tq.update(1)
        eps = max(self.eps_decay ** self.step_num, self.eps_min)
        if self.test:
            eps = 0
        if random() < eps:
            action = self.env.action_space.sample()
        else:
            action = self.model(torch.Tensor(self.last_observation).to(self.device)).max(-1)[-1].item()

        observation, reward, done, info = self.env.step(action)

        self.rolling_reward += reward

        reward = reward / 100.0

        self.rb.insert(SARSD(self.last_observation, action, reward, observation, done))

        if done:
            self.episode_rewards.append(self.rolling_reward)
            if self.test:
                print(self.rolling_reward)
            self.rolling_reward = 0
            observation = self.env.reset()

        self.last_observation = observation

        self.steps_since_train += 1
        self.step_num += 1

        if not self.test and len(
                self.rb.buffer) > self.min_rb_size and self.steps_since_train > self.env_steps_before_train:
            loss = train_step(self.model, self.target, self.rb.sample(self.sample_size), self.env.action_space.n,
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
                print('update tgt model', np.mean(self.episode_rewards))
                self.epochs_since_tgt = 0
                torch.save(self.target.state_dict(), f'models/{self.step_num}.pth')

            self.steps_since_train = 0


def main(test=False, checkpoint=None, device='cuda'):
    game = Game()
    try:
        while True:
            game.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
