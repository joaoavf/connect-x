"""
This is my first MCTS algorithm. It was based on:
https://www.kaggle.com/matant/monte-carlo-tree-search-connectx

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from random import shuffle
import math
from utils import *
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
from time import time
from kaggle_environments import Environment, environments, evaluate
from copy import deepcopy
import ipdb


def ucb1(child, parent_count, exploration_parameter=math.sqrt(2)):
    e1 = math.log(parent_count) / child.count
    return (child.score / child.count) + exploration_parameter * math.sqrt(e1)


class MCTS:
    def __init__(self, model, max_time):
        self.model = model
        self.max_time = max_time

    def return_play(self, current_node):
        t0 = time()

        while time() - t0 < self.max_time:
            self.tree_search(current_node)

        scores = [child.score for child in current_node.children]

        play = current_node.children[scores.index(max(scores))].play

        return transform_play_to_column(play=play)

    def tree_search(self, node):
        if node.value != 0 or node.mask == 279258638311359:  # Find terminal nodes
            node.score += node.value
            node.count += 1
            return -node.value  # Giving higher score to shallow nodes

        # Is new child?
        if node.is_new_child():
            child = node.new_child()
            child.score += child.model_score(model=self.model)
            child.count += 1
            return -child.score

        # Exploit or exploit
        result = self.tree_search(node.explore_or_exploit())

        node.score += int(result == 1)
        node.count += 1

        return -result


class Node:
    def __init__(self, board, bit_board, mask, play=0):
        self.board = board
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = int(connected_four(self.bit_board))
        self.children = []

        self.plays = generate_plays(self.mask)
        self.score, self.count = 0, 1

        shuffle(self.plays)  # Inplace list shuffle

    def is_new_child(self):
        return len(self.plays) > 0

    def explore_or_exploit(self):
        scores = [ucb1(child, self.count, exploration_parameter=math.sqrt(2)) for child in self.children]
        return self.children[scores.index(max(scores))]

    def new_child(self):
        play = self.plays.pop(0)
        new_bit_board = (self.mask ^ self.bit_board) | play
        new_mask = self.mask | play
        new_board = calculate_new_board(self.board, transform_play_to_column(play=play))
        node = Node(board=new_board,
                    bit_board=new_bit_board,
                    mask=new_mask,
                    play=play)
        self.children.append(node)
        return node

    def model_score(self, model):
        return model(torch.Tensor([self.board]))[0].detach().numpy().max()


@dataclass
class SARSD:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    def __init__(self, buffer_size=1_000_000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def insert(self, sarsd):
        self.buffer.append(sarsd)

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)


class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.00001):
        super(Model, self).__init__()
        self.obs_shape, self.num_actions = obs_shape, num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0], 128),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions), torch.nn.Tanh()
        )
        self.initialize_layers()

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def initialize_layers(self):
        # Initialize layers
        for layer in self.net[1:-2:2]:
            torch.nn.init.kaiming_normal_(layer.weight)
        torch.nn.init.normal_(self.net[-2].weight)

    def forward(self, x):
        return self.net(x)


def pre_process(observation):
    board = observation['board']
    player = observation['mark']
    return np.array([1 if value == player else 0 if value == 0 else -1 for value in board])


class Agent:
    def __init__(self, model, max_time):
        self.model = model
        self.target = deepcopy(model)
        self.mcts = MCTS(model=model, max_time=max_time)

    def get_action(self, raw_obs, board):
        np_board = translate_board(raw_obs['board'])
        bit_board, mask = get_position_mask_bitmap(np_board, raw_obs['mark'])

        node = Node(board=board, bit_board=bit_board ^ mask, mask=mask)
        return self.mcts.return_play(current_node=node)

    def update_target_model(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_model_to_disk(self, path):
        torch.save(self.model.state_dict(), path)

    def train_step(self, state_transitions, num_actions, device, discount_factor=0.99):
        cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions]).to(device)
        rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions]).to(device)
        mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]).to(device)
        next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions]).to(device)
        actions = [s.action for s in state_transitions]

        move_validity = next_states[:, :num_actions] == 0
        with torch.no_grad():
            q_values_next = self.target(next_states)
        q_values_next = -np.where(move_validity, q_values_next, -1).max(-1)

        self.model.optimizer.zero_grad()
        qvals = self.model(cur_states)
        one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

        actual_values = rewards[:, 0] + mask[:, 0] * q_values_next * discount_factor

        expected_values = torch.sum(qvals * one_hot_actions, -1)

        loss = ((actual_values - expected_values) ** 2).mean()

        loss.backward()
        self.model.optimizer.step()

        return loss


class ConnectX(Environment):
    def __init__(self):
        super(ConnectX, self).__init__(**environments['connectx'])
        self.action_space = gym.spaces.Discrete(self.configuration.columns)
        self.observation_space = np.array([0] * self.configuration.columns * self.configuration.rows)


class EvolutionaryTrainer:
    def __init__(self, checkpoint=None, device='cpu', num_iters=1_000_000, num_episodes=1_000, pit_freq=100,
                 win_threshold=0.55):
        wandb.init(project="alpha_zero", name="v0")

        self.env = ConnectX()

        model = Model(self.env.observation_space.shape, self.env.action_space.n).to(device)
        if checkpoint is not None:
            print('Models loaded:')
            model.load_state_dict(torch.load(checkpoint))

        self.trainer = Trainer(model)
        self.new_trainer = Trainer(model)
        self.num_iters = num_iters
        self.num_episodes = num_episodes
        self.pit_freq = pit_freq
        self.win_threshold = win_threshold

    def pit_nns(self):
        for i in range(self.num_iters):
            for e in range(self.num_episodes):
                self.trainer.play()

            self.new_trainer.train(self.trainer.rb)
            if i % self.pit_freq == self.pit_freq - 1:
                frac_win = self.pit()  # compare new net with previous net
                if frac_win > self.win_threshold:
                    self.trainer.agent.model = deepcopy(self.new_trainer.agent.model)
                    torch.save(self.new_trainer.agent.model.state_dict(), 'models/az_{i}.pth')
                win_random = self.pit_random()
                update_wandb(frac_win=frac_win, win_random=win_random, step_num=i)

    def pit(self):
        def my_agent(obs, config):
            board = pre_process(observation=obs)
            return self.trainer.agent.get_action(raw_obs=obs, board=board)

        def my_new_agent(obs, config):
            board = pre_process(observation=obs)
            return self.new_trainer.agent.get_action(raw_obs=obs, board=board)

        return get_win_percentages(agent1=my_new_agent, agent2=my_agent, n_rounds=100)

    def pit_random(self):
        def my_new_agent(obs, config):
            board = pre_process(observation=obs)
            return self.trainer.agent.get_action(raw_obs=obs, board=board)

        return get_win_percentages(agent1=my_new_agent, agent2=ConnectX().agents['random'], n_rounds=100)


def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds // 2)
    # Agent 2 goes first (roughly) half the time
    outcomes += [[b, a] for [a, b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds - n_rounds // 2)]
    return np.round(outcomes.count([1, -1]) / len(outcomes), 2)


def update_wandb(frac_win, win_random, step_num):
    wandb.log({'win %': frac_win, 'win_random %': win_random}, step=step_num)


class Trainer:
    def __init__(self, model, env=ConnectX(), device='cpu', min_rb_size=100_000, sample_size=4_096,
                 env_steps_before_train=64, tgt_model_update=250, max_time=0.01):
        self.tq = tqdm()

        self.min_rb_size = min_rb_size
        self.sample_size = sample_size

        self.device = device

        self.env_steps_before_train = env_steps_before_train
        self.tgt_model_update = tgt_model_update

        self.env = env

        self.last_raw_observation = self.env.reset()[0]['observation']
        self.last_observation = pre_process(observation=self.last_raw_observation)

        self.agent = Agent(model=model, max_time=max_time)

        self.rb = ReplayBuffer()
        self.steps_since_train = 0
        self.epochs_since_tgt = 0

        self.step_num = -1 * self.min_rb_size

        self.active_player = 0

    def play(self):

        self.tq.update(1)

        action = self.agent.get_action(raw_obs=self.last_raw_observation, board=self.last_observation)

        observation, reward, done = self.process_action(action)

        self.rb.insert(SARSD(self.last_observation, action, reward, observation, done))

        self.finish_step_routine(observation=observation)

        if done:
            self.new_game()

    def train(self, rb):
        if self.step_num > self.min_rb_size and self.steps_since_train > self.env_steps_before_train:
            self.train_model_routine(rb)

        if self.epochs_since_tgt > self.tgt_model_update:
            self.update_target_model_routine()

    def switch_active_player(self):
        self.active_player = [1, 0][self.active_player]

    def process_action(self, action):
        p_dict = self.env.step([action if i == self.active_player else None for i in [0, 1]])

        reward = p_dict[self.active_player]['reward']
        done = self.env.done

        self.last_raw_observation = p_dict[[1, 0][self.active_player]]['observation']
        observation = pre_process(observation=self.last_raw_observation)

        reward = 1 if reward == 1 else 0
        return observation, reward, done

    def new_game(self):
        self.last_raw_observation = self.env.reset()[0]['observation']
        self.last_observation = pre_process(observation=self.last_raw_observation)
        self.active_player = 0

    def update_target_model_routine(self):
        self.agent.update_target_model()
        self.agent.save_model_to_disk(f'models/target_{self.step_num}.pth')
        self.epochs_since_tgt = 0

    def train_model_routine(self, rb):
        self.epochs_since_tgt += 1
        self.steps_since_train = 0
        loss = self.agent.train_step(rb.sample(self.sample_size), self.env.action_space.n, self.device)

    def finish_step_routine(self, observation):
        self.switch_active_player()
        self.last_observation = observation
        self.steps_since_train += 1
        self.step_num += 1


if __name__ == '__main__':
    et = EvolutionaryTrainer()  # (checkpoint='models/dqn_minimax_1770935.pth')
    et.pit_nns()
