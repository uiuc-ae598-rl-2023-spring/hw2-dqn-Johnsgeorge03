#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:35:09 2023

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

import discreteaction_pendulum
from agent_network import DQN

class Plotter:
    def __init__(self, env, weight_file, hidden_size, device):
        self.env = env
        self.NN  = DQN(env.num_states, env.num_actions, hidden_size).to(device)
        self.NN.load_state_dict(torch.load(weight_file))
        self.device = device
        self.simulate()
    
    def simulate(self):
        self.n        = 100
        self.theta    = torch.linspace(-np.pi, np.pi, self.n)
        self.thetadot = torch.linspace(-self.env.max_thetadot, self.env.max_thetadot, 
                                  self.n)
        state         = torch.cat((self.theta, self.thetadot)).reshape(self.n, 2)\
                        .to(self.device)
        
        self.tau      = np.zeros((self.n, self.n))
        self.q_vals   = np.zeros((self.n, self.n))
        
        with torch.no_grad():
            for i in range(self.n):
                for j in range(self.n):
                    state             = torch.tensor([self.theta[j], 
                                                      self.thetadot[i]])
                    out               = self.NN(state).detach().numpy()
                    self.tau[i, j]    = self.env._a_to_u(np.argmax(out))
                    self.q_vals[i, j] = np.max(out)
        
        
                
        
    def policy(self, state):
        state  = torch.tensor(state, dtype=torch.float32, device= self.device)\
                 .unsqueeze(0)
        with torch.no_grad():
            action = self.NN(state).max(1)[1].view(1, 1).item()
        return action
    
    def plot_torque_vs_vel(self, dest_file):
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        idx    = int(self.n/2)
        torque = self.tau[:, idx]
        theta  = round(float(self.theta[idx]), 2)
        plt.figure(figsize = (5, 5))
        plt.plot(self.thetadot.numpy(), torque, 'r', lw = 2, 
                 label=r'$\theta$ = {}'.format(theta))
        plt.xlabel(r"Angular velocity, $\dot \theta$")
        plt.ylabel(r"Torque, $\tau$")
        plt.legend()
        # plt.tight_layout()
        plt.grid("True")
        plt.savefig(dest_file)
        plt.show()
        
    def plot_learning_curve(self, gamma, source_file, dest_file):
        with open(source_file) as f:
            lines = f.readlines()

        # Remove any leading or trailing whitespace characters from each line
        lines = [line.strip() for line in lines]

        # Split each line into a list of integers
        episode_list = [[int(num) for num in line[1:-1].split(',')] 
                        for line in lines]
        num_eps              = len(episode_list)
        discounted_returns   = np.zeros(num_eps)
        undiscounted_returns = np.zeros(num_eps)
        for e in range(num_eps):
            returns   = 0
            u_returns = 0
            for i in range(len(episode_list[e])):
                returns   += (gamma**i) * episode_list[e][i]
                u_returns += episode_list[e][i]
            discounted_returns[e]   = returns
            undiscounted_returns[e] = u_returns
        
        mean_ud_returns = []
        sd_ud_returns   = []
        
        mean_d_returns  = []
        sd_d_returns    = []
        for i in range(num_eps):
            if i < 100:
                mean_ud_returns.append(np.mean(undiscounted_returns[0:i+1]))
                sd_ud_returns.append(np.std(undiscounted_returns[0:i+1]))
                
                mean_d_returns.append(np.mean(discounted_returns[0:i+1]))
                sd_d_returns.append(np.std(discounted_returns[0:i+1]))
                
            else:
                mean_ud_returns.append(np.mean(undiscounted_returns[i-100:i]))
                sd_ud_returns.append(np.std(undiscounted_returns[i-100:i]))
                
                mean_d_returns.append(np.mean(discounted_returns[i-100:i]))
                sd_d_returns.append(np.std(discounted_returns[i-100:i]))
                
        
        
        fig, ax   = plt.subplots(2, 1, sharex=True, figsize=(10, 9))
        sns.set(style="ticks")
        sns.set_style("darkgrid")
     
        row_means = np.array(mean_d_returns)
        row_stds  = np.array(sd_d_returns)

        # create a pandas DataFrame with columns for x-axis, y-axis, lower bound, and upper bound
        df = pd.DataFrame({'x': range(num_eps),
                            'y': row_means,
                            'lower': row_means - row_stds,
                            'upper': row_means + row_stds})

        # plot the mean values with a variance band using Seaborn's lineplot
        sns.lineplot(data=df, x='x', y='y', ci='sd', linewidth = 2, color='r',
                     ax = ax[0], label = 'Discounted')

        # plot the variance band as a shaded area
        ax[0].fill_between(df['x'], df['lower'], df['upper'], alpha=0.2)
        ax[0].grid("True")
        ax[0].legend(loc="best")

        
        row_means = np.array(mean_ud_returns)
        row_stds  = np.array(sd_ud_returns)

        # create a pandas DataFrame with columns for x-axis, y-axis, lower bound, and upper bound
        df = pd.DataFrame({'x': range(num_eps),
                            'y': row_means,
                            'lower': row_means - row_stds,
                            'upper': row_means + row_stds})

        # plot the mean values with a variance band using Seaborn's lineplot
        sns.lineplot(data=df, x='x', y='y', ci='sd', linewidth = 2, color='r',
                      ax = ax[1], label = 'Uniscounted')
    
        # plot the variance band as a shaded area
        ax[1].fill_between(df['x'], df['lower'], df['upper'], alpha=0.2)
        ax[1].grid("True")
        ax[1].legend(loc="best")
        # show the plot
        ax[0].set_ylim(bottom=-15, top=20)
        ax[0].set_ylabel('Mean Discounted Returns')
        ax[0].set_xlabel('Episodes')
        ax[1].set_ylim(bottom=-110, top=110)
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Mean Undiscounted Returns')
        
        plt.legend(loc="best")
        plt.suptitle("DQN Learning Curve")
        plt.tight_layout()
        plt.savefig(dest_file)
        plt.show()
                
            
            
        
    
    def plot_policy(self, filename):
        x, y = np.meshgrid(self.theta, self.thetadot)
        z    = self.tau
        z_min, z_max = np.min(z), np.max(z)
        plt.figure(figsize = (10, 8))
        plt.xlabel(r"$\theta$", fontsize = 15)
        plt.ylabel(r"$\dot \theta$", fontsize = 15)
        c = plt.pcolormesh(x, y, z, vmin=z_min, vmax=z_max)
        cbar = plt.colorbar(c)
        cbar.ax.tick_params(labelsize=8) 
        cbar.ax.set_ylabel('Tau',fontsize=15)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        
    
    def plot_value_fn(self, filename):
        x, y = np.meshgrid(self.theta, self.thetadot)
        z    = self.q_vals
        z_min, z_max = np.min(z), np.max(z)
        plt.figure(figsize = (10, 8))
        plt.xlabel(r"$\theta$", fontsize = 15)
        plt.ylabel(r"$\dot \theta$", fontsize = 15)
        c = plt.pcolormesh(x, y, z, vmin=z_min, vmax=z_max)
        cbar = plt.colorbar(c)
        cbar.ax.tick_params(labelsize=8) 
        cbar.ax.set_ylabel('Value function',fontsize=15)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        
    
    def generate_video(self, filename):
        self.env.video(self.policy, filename=filename)
    
    def plot_trajectory(self, filename1, filename2):
        s = self.env.reset()

        # Create dict to store data from simulation
        data = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }

        # Simulate until episode is done
        done = False
        while not done:
            a = self.policy(s)
            (s, r, done) = self.env.step(a)
            data['t'].append(data['t'][-1] + 1)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)

        # Parse data from simulation
        data['s'] = np.array(data['s'])
        theta = data['s'][:, 0]
        thetadot = data['s'][:, 1]
        tau = [self.env._a_to_u(a) for a in data['a']]

        # Plot data and save to png file
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(3, 1, sharex = True, figsize=(10, 10))
        ax[0].plot(data['t'], theta, label=r'$\theta$')
        ax[0].plot(data['t'], thetadot, label=r'$\dot \theta$')
        ax[0].grid("True")
        ax[0].legend(fontsize=12)
        
        ax[1].plot(data['t'][:-1], tau, label=r'$\tau$')
        ax[1].grid("True")
        ax[1].legend(fontsize=12)
        
        ax[2].plot(data['t'][:-1], data['r'], label='Reward')
        ax[2].grid("True")
        ax[2].legend(fontsize=12)
        
        ax[2].set_xlabel('Time step')
        plt.tight_layout()
        plt.savefig(filename1)
        plt.show()
        
        x, y = np.meshgrid(self.theta, self.thetadot)
        z    = self.q_vals
        z_min, z_max = np.min(z), np.max(z)
        plt.figure(figsize = (10, 8))
        plt.xlabel(r"$\theta$", fontsize = 15)
        plt.ylabel(r"$\dot \theta$", fontsize = 15)
        c = plt.pcolormesh(x, y, z, vmin=z_min, vmax=z_max)
        plt.plot(theta, thetadot, 'k', label = "Trajectory", lw = 2)
        cbar = plt.colorbar(c)
        cbar.ax.tick_params(labelsize=8) 
        cbar.ax.set_ylabel('Value function',fontsize=15)
        plt.tight_layout()
        plt.legend()
        plt.savefig(filename2)
        plt.show()
        
    
    def plot_ablation_study(self, gamma, file_list, legend_list, dest_file):
        fig, ax   = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        num_files = len(file_list)
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        for i in range(num_files):
            source_file = file_list[i]
            with open(source_file) as f:
                lines = f.readlines()

            # Remove any leading or trailing whitespace characters from each line
            lines = [line.strip() for line in lines]

            # Split each line into a list of integers
            episode_list = [[int(num) for num in line[1:-1].split(',')] 
                            for line in lines]
            num_eps              = len(episode_list)
            discounted_returns   = np.zeros(num_eps)
            undiscounted_returns = np.zeros(num_eps)
            for e in range(num_eps):
                returns   = 0
                u_returns = 0
                for j in range(len(episode_list[e])):
                    returns   += (gamma**j) * episode_list[e][j]
                    u_returns += episode_list[e][j]
                discounted_returns[e]   = returns
                undiscounted_returns[e] = u_returns
            
            mean_ud_returns = []
            sd_ud_returns   = []
            
            mean_d_returns  = []
            sd_d_returns    = []
            for j in range(num_eps):
                if j < 100:
                    mean_ud_returns.append(np.mean(undiscounted_returns[0:j+1]))
                    sd_ud_returns.append(np.std(undiscounted_returns[0:j+1]))
                    
                    mean_d_returns.append(np.mean(discounted_returns[0:j+1]))
                    sd_d_returns.append(np.std(discounted_returns[0:j+1]))
                    
                else:
                    mean_ud_returns.append(np.mean(undiscounted_returns[j-100:j]))
                    sd_ud_returns.append(np.std(undiscounted_returns[j-100:j]))
                    
                    mean_d_returns.append(np.mean(discounted_returns[j-100:j]))
                    sd_d_returns.append(np.std(discounted_returns[j-100:j]))
                    
            
            
         
            row_means = np.array(mean_d_returns)
            row_stds  = np.array(sd_d_returns)

            # create a pandas DataFrame with columns for x-axis, y-axis, lower bound, and upper bound
            df = pd.DataFrame({'x': range(num_eps),
                                'y': row_means,
                                'lower': row_means - row_stds,
                                'upper': row_means + row_stds})

            # plot the mean values with a variance band using Seaborn's lineplot
            sns.lineplot(data=df, x='x', y='y', ci='sd',
                         label = legend_list[i], ax = ax[0], linewidth=3)
            # plot the variance band as a shaded area
            ax[0].fill_between(df['x'], df['lower'], df['upper'], alpha=0.2)
            ax[0].grid("True")
            ax[0].legend(loc="best")
            # set the plot labels and legend
            
            

            
            row_means = np.array(mean_ud_returns)
            row_stds  = np.array(sd_ud_returns)
            # create a pandas DataFrame with columns for x-axis, y-axis, lower bound, and upper bound
            df = pd.DataFrame({'x': range(num_eps),
                                'y': row_means,
                                'lower': row_means - row_stds,
                                'upper': row_means + row_stds})

            # plot the mean values with a variance band using Seaborn's lineplot
            sns.lineplot(data=df, x='x', y='y', ci='sd', 
                         label = legend_list[i], ax = ax[1], linewidth=3)

            # plot the variance band as a shaded area
            ax[1].fill_between(df['x'], df['lower'], df['upper'], alpha=0.2)
            ax[1].grid("True")
            ax[1].legend(loc="best")
            # set the plot labels and legend
            
           
        ax[0].set_ylim(bottom=-15, top=20)
        ax[0].set_ylabel('Mean Discounted Returns')
        ax[0].set_xlabel('Episodes')
        ax[1].set_ylim(bottom=-110, top=110)
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Mean Undiscounted Returns')
        plt.legend(loc="best")
        plt.suptitle("DQN Learning Curve")
        plt.tight_layout()
        plt.savefig(dest_file)
        plt.show()
        
    def plot_ablation_study_bar(self, num_eps, gamma, weight_list, dest_file):
        
        x_list          = ['with target, with replay', 
                          'with target, without replay',
                          'without target, with replay', 
                          'without target, without replay']
        max_mean_ud     = []
        max_mean_d      = []
        w_size          = 100
        for weight_file in weight_list:
            self.NN.load_state_dict(torch.load(weight_file))
            un_returns_list = []
            d_returns_list  = []
            for eps in range(num_eps):
                done       = False
                s          = self.env.reset()
                d_returns  = 0.0
                ud_returns = 0.0
                count      = 0
                while not done:
                    a             = self.policy(s)
                    (s, r, done)  = self.env.step(a)
                    d_returns    += (gamma ** count) * r
                    ud_returns   += r
                    count        += 1 
                un_returns_list.append(ud_returns)
                d_returns_list.append(d_returns)
                
            # compute runnig mean of returns for 10 eps
            max_mean_ud.append(np.max(np.convolve(un_returns_list, 
                                           np.ones(w_size)/w_size, 
                                           mode='valid')))
            max_mean_d.append(np.max(np.convolve(d_returns_list, 
                                           np.ones(w_size)/w_size, 
                                           mode='valid')))
            
                
            
            
        fig, ax   = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        sns.barplot(x=x_list,y=max_mean_d, ax=ax[0])
        sns.barplot(x=x_list,y=max_mean_ud, ax=ax[1])
        ax[0].set_ylabel('Mean Discounted Returns')
        ax[1].set_ylabel('Mean Undiscounted Returns')
        plt.suptitle("Maximum mean episode reward")
        plt.tight_layout()
        plt.savefig(dest_file)
        plt.show()
        
        

