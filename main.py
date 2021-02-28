import numpy as np
import torch
import gym

from optimizer import Optimizer
from utils.storage_ddpg import Storage_ddpg
from utils.storage_dqn import Storage_dqn
from utils.goal import Goal
from env import MyEnv
from policies.ddpg import Actor, Critic
from policies.dqn import DQN

n_episodes = 200
learning_rate = 1e-3
n_steps = 500
max_grad_norm = 0.5
update_epochs = 1
discount = 0.99

mini_batch_size_dqn = 256
buffer_size_dqn = 2048

mini_batch_size_ddpg = 256
buffer_size_ddpg = 2048

var = 1
e_decay = 0.99998
e_meta_decay = 0.02

target_policy_update = 5

num_goals = 7
num_hidden_1 = 64
num_hidden_2 = 32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_intrinsic_reward(goal, state, env):
    reward = 0
    if 1 <= goal <= 7:
        idx = goal
        if np.sqrt(np.sum((env.sensor[idx-1].pos - state[:2]) ** 2)) <= 40:
            reward = 10
    else:
        if np.sqrt(np.sum((env.BS.pos - state[:2]) ** 2)) <= 80:
            reward = 10
    return reward

def get_policies(env):
    meta_policy = DQN(env.observation_space.shape[0], num_goals, num_hidden_1, num_hidden_2)
    target_meta_policy = DQN(env.observation_space.shape[0], num_goals, num_hidden_1, num_hidden_2)

    policy_actor = Actor(2, env.action_space.shape[0], num_hidden_1, num_hidden_2)
    target_policy_actor = Actor(2, env.action_space.shape[0], num_hidden_1, num_hidden_2)
    policy_critic = Critic(2, env.action_space.shape[0], num_hidden_1, num_hidden_2)
    target_policy_critic = Critic(2, env.action_space.shape[0], num_hidden_1, num_hidden_2)

    return meta_policy, target_meta_policy, policy_actor, target_policy_actor, policy_critic, target_policy_critic

def main():
    env = MyEnv()

    goal_object = Goal()
    meta_policy, target_meta_policy, policy_actor, target_policy_actor, policy_critic, target_policy_critic = get_policies(env)
    optimizer = Optimizer(meta_policy, target_meta_policy, policy_actor, target_policy_actor, policy_critic, target_policy_critic, mini_batch_size, discount, learning_rate, update_epochs)

    episode_rewards = []

    frame = 0
    meta_frame = 0

    Buffer_DQN = Storage_dqn(buffer_size_dqn)
    Buffer_DDPG = Storage_ddpg(buffer_size_ddpg)

    get_meta_epsilon = lambda episode: np.exp(-episode * e_meta_decay)
    get_epsilon = lambda episode: np.power()

    for episode in range(0, n_episodes+1):
        meta_state = env.reset()
        state = meta_state[:2]

        done = False

        for step_1 in range(200):
            goal = meta_policy.act(meta_state, get_meta_epsilon(meta_frame))
            one_hot_goal = Goal.get_one_hot_goal(goal)

            goal_reached = False
            extrinsic_reward = 0

            for step_2 in range(200):

                joint_state = torch.cat((torch.FloatTensor(state), one_hot_goal), axis=0)
                action = policy_actor.act(joint_state)
                next_state, reward, done, info = env.step(action)
                intrinsic_reward = get_intrinsic_reward(goal, state, env)
                goal_reached = True if intrinsic_reward != 0 else False
                next_joint_state = np.concatenate([next_state[:2], one_hot_goal.detach().numpy()], axis=0)

                Buffer_DDPG.add(joint_state, action, intrinsic_reward, next_joint_state, done)

                if goal == 0:
                    extrinsic_reward += reward

                state = next_joint_state

                frame += 1

                if done or goal_reached:
                    break

            Buffer_DDPG.add(meta_state, goal, extrinsic_reward, next_state, done)
            meta_state =

