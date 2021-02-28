import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
tf.keras.backend.set_floatx('float64')

class SpgAgent(keras.models.Model):

    def __init__(self, SPG_network, training_param, data_logger, buffer):
        super(SpgAgent, self).__init__()
        self.model = SPG_network
        self.training_param = training_param
        self.data_logger = data_logger
        self.buffer = buffer
    

    def train_step(self):
        """
        Train agent per episode if not using training replay buffer and per timestep if using the replay buffer.
        """
        use_buffer = self.training_param["use_replay_buffer"]
        batch_size = self.training_param["batch_size"]
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        gamma = self.training_param["gamma"]

        if use_buffer:
            # Sample mini-batch from memory
            batch = self.buffer.get_training_samples(batch_size)
            states = tf.squeeze(tf.convert_to_tensor([each[0] for each in batch]))
            actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in batch])))
            rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in batch])))
            next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in batch])))
            # TODO check using states or next_states in other repo
        else:
            # actions correspond to actions given to the simulator [0,1]
            timesteps, states, rewards, actions = self.data_logger.get_experience()
            batch_size = len(timesteps)


        with tf.GradientTape() as tape:
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            # Return = SUM_t=0^inf (gamma*reward_t)

            action_probs, critic_value = self.model(states)

            chosen_action_prob = []
            for t in range(batch_size):
                chosen_action_prob.append(action_probs[t, actions[t]])
            chosen_action_prob = tf.convert_to_tensor(chosen_action_prob)
            actor_log_prob = tf.math.log(chosen_action_prob)

            returns = []
            discounted_sum = 0
            for r in rewards[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()
            self.data_logger.returns = returns

            # Calculating loss values to update our network
            history = zip(actor_log_prob, critic_value, returns)
            actor_losses = []
            critic_losses = []
            for actor_log_prob, crit_value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up receiving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - crit_value
                self.data_logger.advantage.append(diff)
                actor_losses.append(-actor_log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    self.training_param["loss_func"](tf.expand_dims(crit_value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            self.data_logger.losses.append(loss_value)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.data_logger.gradients.append(grads)
            self.training_param["optimiser"].apply_gradients(zip(grads, self.model.trainable_variables))
            # Clear the loss and reward history



