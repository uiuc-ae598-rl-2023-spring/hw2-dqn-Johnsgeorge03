#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:36:53 2023

@author: john
"""

from agent_network import *
import Plotter

import matplotlib.pyplot as plt
import discreteaction_pendulum
import seaborn as sns
import pandas as pd
import torch
import os 

# def main():
plt.ion()
## ENVIRONMENT

env                 = discreteaction_pendulum.Pendulum()
n_observations      = env.num_states
n_actions           = env.num_actions

## HYPERPARAMETERS
hidden_size         = 64
gamma               = 0.95

learning_rate       = 0.0001
epsilon_start       = 0.9
epsilon_end         = 0.05
epsilon_decay       = 1000
tau                 = 1.0

## AGENT
device              = ""
if torch.backends.mps.is_available():
            device  = "mps"
elif torch.cuda.is_available():
            device  = "cuda"
else:
            device  = "cpu"
            
device              = "cpu" # turns out mps is so slow 
agent               = Agent(n_observations, n_actions, hidden_size, 
                            learning_rate, gamma, epsilon_start, epsilon_end, 
                            epsilon_decay, tau, device)

agent.batch_size    = 128
agent.memory_size   = 10000 # if equal to batch_size implies no replay
agent.update_freq   = 500 # > 1 implies target network
agent.anneal_steps  = 10000
# update freq = 500
# no replay batch size = one-tenth of total steps(10000)

## DIRECTORIES
weight_dir          = 'NN_weights/target_replay/'
fig_dir             = 'figures/target_replay/'
reward_dir          = 'rewards/'
reward_file         = 'rewards.txt'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(reward_dir, exist_ok=True)

## TRAIN
num_episodes        = 1000
scores              = []
mean_scores         = []
max_mean            = 0
all_udissc_r        = []

for i in range(num_episodes):
    s               = env.reset()
    s               = torch.tensor(s, dtype=torch.float32, device=agent.device)\
                      .unsqueeze(0)
    done            = False
    total_reward    = 0
    rewards_list    = []
    count           = 0
    while not done:
        action      = agent.choose_action(s) 
        assert(action.item() in np.arange(n_actions))
        obs, r, done = env.step(action.item())
        rewards_list.append(r)
        total_reward += r
        if done:
            s1  = None
        else:
            s1  = torch.tensor(obs, dtype = torch.float32, device = device) \
                 .unsqueeze(0)
        r        = torch.tensor([r], device = agent.device) 
        agent.memory.push(s, action, s1 , r)
        s        = s1
        count   += 1
        agent.replay_and_learn()
        if agent.steps % agent.update_freq == 0:
            agent.update_target_network()
        

    all_udissc_r.append(rewards_list)
    scores.append(total_reward)
    agent.plot_rewards(scores)
    mean = np.mean(scores[-100:])
    mean_scores.append(mean)
    if mean > max_mean:
        torch.save(agent.policy_network.state_dict(), 
                    weight_dir + 'qnet_model_weights_max_mean.pth')
        torch.save(agent.target_network.state_dict(), 
                    weight_dir + 'tnet_model_weights_max_mean.pth')
        max_mean = mean
        
    print('Episode {}, Total iterations {}, Total Reward {:.2f}, Mean Reward {:.2f}, Epsilon: {:.2f}'
          .format(i + 1, agent.steps,  
          total_reward, np.mean(scores[-100:]), agent.epsilon))



print("COMPLETE")
agent.plot_rewards(scores, show_result = True)
plt.ioff
plt.show()


## WRITE REWARDS AND MODEL PARAMS TO FILES
torch.save(agent.policy_network.state_dict(), weight_dir + 'qnet_model_weights_end.pth')
torch.save(agent.target_network.state_dict(), weight_dir + 'tnet_model_weights_end.pth')
with open(reward_dir + reward_file, 'w+') as f:
    for items in all_udissc_r:
        f.write('%s\n' %items)
    print("File written successfully")
f.close()

# weight_dir          = 'NN_weights/target_replay/'
# fig_dir             = 'figures/target_replay/'
# reward_dir          = 'rewards/'
# reward_file         = 'rewards.txt'


## PLOTS
weight_file   = weight_dir + 'qnet_model_weights_end.pth'
plotter       = Plotter.Plotter(env, weight_file, hidden_size, device)
plotter.plot_policy(fig_dir + 'policy.png')
plotter.plot_value_fn(fig_dir + 'value_fn.png')
plotter.plot_trajectory(fig_dir + 'trajectory.png')
plotter.generate_video(fig_dir + 'trajectory_video.gif')
plotter.plot_learning_curve(gamma, reward_dir + reward_file, 
                            fig_dir + 'undiscounted_learning_curve.png', 
                            fig_dir + 'discounted_learning_curve.png')

legend_list   = ['returns', 'returns (no replay)', 'returns (no target)',
                  'returns (no replay, no target)']
file_list     = [reward_dir + 'rewards.txt', reward_dir + 'rewards_no_rep.txt', 
                 reward_dir + 'rewards_no_tar.txt',
                 reward_dir + 'rewards_no_tar_no_rep.txt']

plotter.plot_ablation_study(gamma, file_list, legend_list, 
                            'figures/ablation_study.png',
                            'figures/ablation_study_bar.png')


# if __name__ == '__main__':
#     main()