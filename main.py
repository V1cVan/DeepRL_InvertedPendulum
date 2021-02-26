import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

from CartPole import CartPoleEnv # CartPoleEnv has been modified to make it more difficult.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# tf.config.experimental.set_visible_devices([], "GPU")

from Policies import *
from Logging import *


class Main(object):
    def __init__(self, env, trainer, data_logger, buffer):
        self.env = env
        self.trainer = trainer
        self.data_logger = data_logger
        self.buffer = buffer

    def train(self):
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
            self.buffer.states.append(state)
            episode_reward = 0
            self.buffer.episode = episode_count
            for timestep in range(max_timesteps):
                self.buffer.timesteps.append(timestep)
                if episode_count % 50 == 0:
                    self.env.render()


                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = self.trainer.model(state)
                # actor calculates the probabilities of outputs from the given state
                # critic evaluates the action by computing the value function

                self.buffer.actions.append(action_probs)
                self.buffer.critic.append(critic_value[0, 0])

                # Sample action from action probability distribution
                # squeeze removes single-dimensional entries from array shape
                # choose between 0 and 1 with the probabilities calculated from network
                action = np.random.choice(self.trainer.model.model_params["num_outputs"],
                                          p=np.squeeze(action_probs))
                self.buffer.chosen_actions.append(action)
                self.buffer.chosen_action_log_prob.append(tf.math.log(action_probs[0, action]))
                # log probabilities have better accuracy (better represent small probabilities)

                # Apply the sampled action in our environment
                state, reward, done, info = env.step(action)
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                self.buffer.states.append(state)
                self.buffer.rewards.append(reward)
                episode_reward += reward

                if done:
                    break

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward_history.append(running_reward)

            self.buffer.states = tf.squeeze(self.buffer.states)
            self.buffer.actions = tf.squeeze(self.buffer.actions)
            self.buffer.chosen_action_log_prob = tf.squeeze(self.buffer.chosen_action_log_prob)
            self.buffer.critic = tf.squeeze(self.buffer.critic)

            self.trainer.train_step()
            self.data_logger.add_episode_data(self.buffer.get_buffer())
            self.buffer.clear_buffer()

            # Log details
            episode_count += 1
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode_count))



            if running_reward > 800 or episode_count >= self.trainer.training_param["max_num_episodes"]:
                print("Solved at episode {}!".format(episode_count))
                self.data_logger.plot_training_data(plot_items)
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
        "optimiser": keras.optimizers.Adam(learning_rate=0.001),
        "loss_func": keras.losses.Huber(),
    }

    model_param = {
        "seed": seed,
        "num_inputs": 4,
        "num_outputs": 2,
        "num_neurons": [200, 150],
        "af": "elu",
        "weights_file_loc": "./model/model_weights"
    }

    env = CartPoleEnv()  # Create the environment
    env.seed(training_param["seed"])
    tf.random.set_seed(training_param["seed"])
    np.random.seed(training_param["seed"])

    SPG_network = SpgNetwork(model_param)
    # SPG_network.display_model_overview()
    buffer = TrainingBuffer()
    SPG_trainer = SpgTrainer(SPG_network, training_param, buffer)

    data_logger = DataLogger(training_param["seed"])

    main = Main(env, SPG_trainer, data_logger, buffer)

    # main.train()

    main.runSimulation(1000)
