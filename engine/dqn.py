from collections import deque

import gym
import ipdb
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


def train_step(model, target, state_transitions, num_actions):
    cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions])
    rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions])
    mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])
    next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions])
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        q_vals_next = target(next_states).max(-1)[0]

    model.optim.zero_grad()
    qvals = model(cur_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)

    loss = ((rewards + mask[:, 0] * q_vals_next - torch.sum(qvals * one_hot_actions, -1)) ** 2).mean()
    loss.backward()
    model.optim.step()

    return loss


def update_target_model(model, target):
    target.load_state_dict(model.state_dict())


def main(test=False, checkpoint=None):
    if not test:
        wandb.init(project="dqn-tutorial", name="dqn-cartpole")
    min_rb_size = 10000
    sample_size = 2500

    eps_min = 0.01
    eps_decay = 0.999999

    env_steps_before_train = 100
    tgt_model_update = 500

    env = gym.make('CartPole-v1')
    last_observation = env.reset()

    m = Model(env.observation_space.shape, env.action_space.n)
    if checkpoint is not None:
        m.load_state_dict(torch.load(checkpoint))
    target = Model(env.observation_space.shape, env.action_space.n)

    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0

    step_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.01)
            tq.update(1)
            eps = max(eps_decay ** step_num, eps_min)
            if test:
                eps = 0
            if random() < eps:
                action = env.action_space.sample()
            else:
                action = m(torch.Tensor(last_observation)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)

            rolling_reward += reward

            reward = reward / 100.0

            rb.insert(SARSD(last_observation, action, reward, observation, done))

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0
                observation = env.reset()

            last_observation = observation

            steps_since_train += 1
            step_num += 1

            if not test and len(rb.buffer) > min_rb_size and steps_since_train > env_steps_before_train:
                loss = train_step(m, target, rb.sample(sample_size), env.action_space.n)
                wandb.log({'loss': loss, 'eps': eps, 'rewards': np.mean(episode_rewards)}, step=step_num)

                episode_rewards = []
                epochs_since_tgt += 1

                if epochs_since_tgt > tgt_model_update:
                    update_target_model(m, target)
                    print('update tgt model', np.mean(episode_rewards))
                    epochs_since_tgt = 0
                    torch.save(target.state_dict(), f'models/{step_num}.pth')

                steps_since_train = 0

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
