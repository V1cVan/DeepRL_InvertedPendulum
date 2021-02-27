import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

from CartPole import CartPoleEnv  # CartPoleEnv has been modified to make it more difficult.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# tf.config.experimental.set_visible_devices([], "GPU")

from Policies import *
from Logging import *
from Models import *

class Main(object):
    def __init__(self, env, trainer, data_logger, buffer):
        self.env = env
        self.trainer = trainer
        self.data_logger = data_logger
        self.buffer = buffer

    def train_SPG(self):
        rewards_history = []
        running_reward_history = []
        running_reward = 0
        episode_count = 0
        max_timesteps = self.trainer.training_param["max_timesteps"]
        # Run until all episodes completed (reward level reached)
        plot_items = self.data_logger.init_training_plot()
        while True:
            # Set environment with random state array: X=(x,xdot,theta,theta_dot)
            state = self.env.reset()
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            self.data_logger.states.append(state)
            episode_reward = 0
            self.data_logger.episode = episode_count
            for timestep in range(max_timesteps):
                self.data_logger.timesteps.append(timestep)
                if episode_count % 50 == 0:
                    self.env.render()


                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = self.trainer.model(state)
                # actor calculates the probabilities of outputs from the given state
                # critic evaluates the action by computing the value function

                self.data_logger.actions.append(action_probs)
                self.data_logger.critic.append(critic_value[0, 0])

                # Sample action from action probability distribution
                # squeeze removes single-dimensional entries from array shape
                # choose between 0 and 1 with the probabilities calculated from network
                action = np.random.choice(self.trainer.model.model_params["num_outputs"],
                                          p=np.squeeze(action_probs))
                self.data_logger.chosen_actions.append(action)
                self.data_logger.chosen_action_log_prob.append(tf.math.log(action_probs[0, action]))
                # log probabilities have better accuracy (better represent small probabilities)

                # Apply the sampled action in our environment
                state, reward, done, info = env.step(action)
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                self.data_logger.states.append(state)
                self.data_logger.rewards.append(reward)
                episode_reward += reward

                if done:
                    break

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward_history.append(running_reward)

            self.data_logger.states = tf.squeeze(self.data_logger.states)
            self.data_logger.actions = tf.squeeze(self.data_logger.actions)
            self.data_logger.chosen_action_log_prob = tf.squeeze(self.data_logger.chosen_action_log_prob)
            self.data_logger.critic = tf.squeeze(self.data_logger.critic)

            self.trainer.train_step()
            self.data_logger.add_episode_data()
            self.data_logger.clear_episode_data()

            # Log details
            episode_count += 1
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode_count))



            if running_reward >= 850 or episode_count >= self.trainer.training_param["max_num_episodes"]:
                print("Solved at episode {}!".format(episode_count))
                self.data_logger.plot_training_data(plot_items)
                file_loc = self.trainer.model.model_params["weights_file_loc"]
                self.trainer.model.save_weights(file_loc)
                break

    def pre_train(self):
        # Fill training buffer with experiences to start training
        max_timesteps = self.trainer.training_param["max_timesteps"]
        while len(self.buffer.buffer) < self.trainer.training_param["max_buffer_size"]:
            self.env.reset()
            action = self.env.action_space.sample()
            state, reward, done, _ = self.env.step(action)
            for t in range(max_timesteps):
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    next_state = np.zeros(np.shape(state))
                    self.buffer.add_experience((state, action, reward, next_state))
                    state = self.env.reset()
                    break
                else:
                    self.buffer.add_experience((state, action, reward, next_state))
                    state = next_state

        return state

    def train_DQN(self):
        epsilon_start = self.trainer.training_param["epsilon_start"]
        epsilon_stop = self.trainer.training_param["epsilon_end"]
        decay = self.trainer.training_param["decay_rate"]

        rewards_history = []
        running_reward_history = []
        running_reward = 0
        episode_count = 0
        max_timesteps = self.trainer.training_param["max_timesteps"]

        # Run until all episodes completed (reward level reached)
        state = self.pre_train()

        while True:
            episode_reward = 0
            self.data_logger.episode = episode_count
            for timestep in range(max_timesteps):
                self.data_logger.timesteps.append(timestep)
                if episode_count % 50 == 0:
                    self.env.render()

                explore_prob = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay*timestep)
                if explore_prob > np.random.rand():
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.trainer.model(state))

                next_state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                if not done:
                    self.buffer.add_experience((state, action, reward, next_state))
                    state = next_state
                    self.trainer.train_step()
                else:
                    next_state = np.zeros(np.shape(state))
                    self.buffer.add_experience((state, action, reward, next_state))
                    self.env.reset()
                    state, reward, done, _ = env.step(self.env.action_space.sample())
                    self.trainer.train_step()
                    break


            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward_history.append(running_reward)

            # Log details
            episode_count += 1
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode_count))

            if running_reward >= 850 or episode_count >= self.trainer.training_param["max_num_episodes"]:
                print("Solved at episode {}!".format(episode_count))
                file_loc = self.trainer.model.model_params["weights_file_loc"]
                self.trainer.model.save_weights(file_loc)
                break

    def runSimulation(self, simulated_timesteps):
        state = env.reset()
        for _ in range(simulated_timesteps):
            env.render()
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            # action to take based on model:
            file_loc = self.trainer.model.model_params["weights_file_loc"]
            self.trainer.model.load_weights(file_loc)
            action_probabilities, critic_values = self.trainer.model(state)
            action = np.random.choice(self.trainer.model.model_params["num_outputs"],
                                      p=np.squeeze(action_probabilities))
            # get observation from environment based on action taken:
            observation, reward, done, info = env.step(action)
            state = observation
            if done:
                env.close()
                break


if __name__ == "__main__":
    seed = 42
    # Set configuration parameters for the whole setup
    training_param = {
        "seed": seed,
        "gamma": 0.99,
        "max_timesteps": 1000,
        "max_num_episodes": 1000,
        "max_buffer_size": 10000,
        "batch_size": 300,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "decay_rate": 1e-4,
        "optimiser": keras.optimizers.Adam(learning_rate=0.001),
        "loss_func": keras.losses.Huber(),
    }

    model_param = {
        "seed": seed,
        "num_inputs": 4,
        "num_outputs": 2,
        "num_neurons": [100, 100],
        "af": "relu",
        "weights_file_loc": "./model/model_weights"
    }

    env = CartPoleEnv()  # Create the environment
    env.seed(training_param["seed"])
    tf.random.set_seed(training_param["seed"])
    np.random.seed(training_param["seed"])

    data_logger = DataLogger(training_param["seed"])
    buffer = TrainingBuffer(training_param["max_buffer_size"])

    SPG_network = SpgNetwork(model_param)
    # SPG_network.display_model_overview()
    SPG_trainer = SpgTrainer(SPG_network, training_param, data_logger, buffer)

    DQN_network = DQNetwork(model_param)
    DQN_trainer = DqnTrainer(DQN_network, training_param, data_logger, buffer)

    # main = Main(env, SPG_trainer, data_logger, buffer)
    # main.train_SPG()

    main = Main(env, DQN_trainer, data_logger, buffer)
    main.train_DQN()

    main.runSimulation(1000)
