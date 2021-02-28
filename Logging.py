from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from collections import deque


class DataLogger(object):
    """
    Data logging class for debugging and monitoring of training results.
    """

    def __init__(self, seed):
        self.seed = seed
        self.episodes = []
        self.episode = 0
        self.timesteps = []
        self.states = []
        self.actions = []  # Network output
        self.chosen_action_log_prob = []  # Log prob
        self.chosen_actions = []  # To simulator
        self.critic = []
        self.rewards = []
        self.losses = []
        self.advantage = []
        self.gradients = []

    def get_var_sizes(self):
        output_dict = {
            "episode": np.shape(np.array(self.episode)),
            "timestesps": np.shape(np.array(self.timesteps)),
            "states": np.shape(np.array(self.states)),
            "actions": np.shape(np.array(self.actions)),  # Network output
            "chosen_action_log_prob": np.shape(np.array(self.chosen_action_log_prob)),  # Log prob
            "chosen_actions": np.shape(np.array(self.chosen_actions)),  # To simulator
            "critic": np.shape(np.array(self.critic)),
            "rewards": np.shape(np.array(self.rewards)),
            "losses": np.shape(np.array(self.losses)),
            "advantage": np.shape(np.array(self.advantage)),
            "gradients": np.shape(np.array(self.gradients))
        }
        return output_dict

    def get_experience(self):
        return self.timesteps, \
               self.states, \
               self.rewards, \
               self.chosen_actions

    def get_episode_data(self):
        complete_episode = {
            "episode": self.episode,
            "timestesps": self.timesteps,
            "states": self.states,
            "actions": self.actions,  # Network output
            "chosen_action_log_prob": self.chosen_action_log_prob,  # Log prob
            "chosen_actions": self.chosen_actions,  # To simulator
            "critic": self.critic,
            "rewards": self.rewards,
            "losses": self.losses,
            "advantage": self.advantage,
            "gradients": self.gradients
        }
        return complete_episode

    def clear_episode_data(self):
        self.timesteps = []
        self.states = []
        self.actions = []
        self.chosen_action_log_prob = []
        self.chosen_actions = []
        self.critic = []
        self.rewards = []
        self.losses = []
        self.advantage = []
        self.gradients = []

    def add_episode_data(self):
        self.episodes.append(self.get_episode_data())

    def init_training_plot(self):
        fig = plt.figure(0, figsize=(18, 12))

        rewards_graph = fig.add_subplot(221)
        rewards_graph.set_autoscale_on(True)  # enable autoscale
        rewards_graph.autoscale_view(True, True, True)
        r_lines, = rewards_graph.plot([], [], 'r.-')
        plt.title("Total rewards per episode.")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.pause(0.001)
        advantage_graph = fig.add_subplot(222)
        advantage_graph.set_autoscale_on(True)  # enable autoscale
        advantage_graph.autoscale_view(True, True, True)
        a_lines, = advantage_graph.plot([], [], 'b.-')
        plt.title("Average advantage (Temporal Diff.) per episode.")
        plt.xlabel("Episode")
        plt.ylabel("Advantage")
        plt.grid(True)
        plt.pause(0.001)
        losses_graph = fig.add_subplot(223)
        losses_graph.set_autoscale_on(True)  # enable autoscale
        losses_graph.autoscale_view(True, True, True)
        l_lines, = losses_graph.plot([], [], 'g.-')
        plt.title("Average loss (objective scalar) per episode.")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.pause(0.001)
        grad_graph = fig.add_subplot(224)
        grad_graph.set_autoscale_on(True)  # enable autoscale
        grad_graph.autoscale_view(True, True, True)
        g_lines, = grad_graph.plot([], [], '.-')
        plt.title("Average gradient value per set of gradients per episode.")
        plt.xlabel("Episode")
        plt.ylabel("Gradients.")
        plt.grid(True)
        plt.pause(0.001)
        plt.ion()

        lines = [r_lines, a_lines, l_lines, g_lines]
        axes = [rewards_graph, advantage_graph, losses_graph, grad_graph]

        return [fig, axes, lines]

    def plot_training_data(self, plot_items):
        """ Plots the training data."""

        fig, axes, lines = tuple(plot_items)
        rewards_graph, advantage_graph, losses_graph, grad_graph = tuple(axes)
        r_lines, a_lines, l_lines, g_lines = tuple(lines)

        num_episodes = len(self.episodes)
        ep = np.arange(num_episodes)

        # TODO confidence intervals?

        rewards_sum = []
        advantage_mean = []
        losses = []
        gradients_mean = []

        for i in ep:
            rewards_sum.append(np.sum(self.episodes[i]["rewards"]))
            advantage_mean.append(np.mean(self.episodes[i]["advantage"]))
            losses.append(self.episodes[i]["losses"])
            grad_layers_mean = []
            for j in np.arange(len(self.episodes[i]["gradients"][0])):
                grad_layers_mean.append(np.mean(self.episodes[i]["gradients"][0][j]))
            gradients_mean.append(grad_layers_mean)

        fig.canvas.flush_events()

        r_lines.set_data(ep, rewards_sum)
        rewards_graph.relim()  # Recalculate limits
        rewards_graph.autoscale_view(True, True, True)  # Autoscale
        plt.pause(0.001)
        a_lines.set_data(ep, advantage_mean)
        advantage_graph.relim()  # Recalculate limits
        advantage_graph.autoscale_view(True, True, True)  # Autoscale
        plt.pause(0.001)
        l_lines.set_data(ep, losses)
        losses_graph.relim()  # Recalculate limits
        losses_graph.autoscale_view(True, True, True)  # Autoscale
        plt.pause(0.001)
        gradients_mean = np.array(gradients_mean)
        for i in range(len(gradients_mean[0])):
            g_lines.set_data(ep, gradients_mean[:, i])
            grad_graph.relim()  # Recalculate limits
            grad_graph.autoscale_view(True, True, True)  # Autoscale
        plt.pause(0.001)
        plt.draw()
        plt.savefig('Training_plots.png')


class TrainingBuffer(object):
    """
    The training buffer is used to store experiences that are then sampled from uniformly to facilitate
    improved training. The training buffer reduces the correlation between experiences and avoids that
    the network 'forgets' good actions that it learnt previously.
    """

    def __init__(self, max_mem_size):
        self.buffer = deque(maxlen=max_mem_size)

    def add_experience(self, experience):
        """
        Add an experience (s_k, a_k, r_k, s_k+1) to the training buffer.
        """
        self.buffer.append(experience)

    def get_training_samples(self, batch_size):
        index = np.random.choice(np.arange(len(self.buffer)),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]
