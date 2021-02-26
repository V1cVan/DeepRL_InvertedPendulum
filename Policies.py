import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
tf.keras.backend.set_floatx('float64')

# class SpgNetwork(keras.Model):
#     def __init__(self, model_param):
#         super(SpgNetwork, self).__init__()
#         self.model_params = model_param
#
#         num_inputs = self.model_params["num_inputs"]
#         num_outputs = self.model_params["num_outputs"]
#         num_hidden_1 = self.model_params["num_neurons"][0]
#         num_hidden_2 = self.model_params["num_neurons"][1]
#         af = self.model_params["af"]
#
#         inputs = layers.Input(shape=(num_inputs,))
#
#         # Actor net
#         dense_actor_1 = layers.Dense(250, activation=af)(inputs)
#         actor = layers.Dense(num_outputs, activation="softmax")(dense_actor_1)
#
#         # Critic net
#         critic = layers.Dense(1)(dense_actor_1)
#
#         self.model = keras.Model(inputs=inputs, outputs=[actor, critic])
#
#     def call(self, input_state):
#         y = self.model(input_state)
#         return y

class SpgNetwork(keras.Model):
    def __init__(self, model_param):
        super(SpgNetwork, self).__init__()
        self.model_params = model_param

        num_inputs = self.model_params["num_inputs"]
        num_outputs = self.model_params["num_outputs"]
        num_hidden_1 = self.model_params["num_neurons"][0]
        num_hidden_2 = self.model_params["num_neurons"][1]
        af = self.model_params["af"]

        inputs = layers.Input(shape=(num_inputs,))

        # Actor net
        dense_actor_1 = layers.Dense(num_hidden_1, activation=af)(inputs)
        dense_actor_2 = layers.Dense(num_hidden_2, activation=af)(dense_actor_1)
        actor = layers.Dense(num_outputs, activation="softmax")(dense_actor_2)

        # Critic net
        dense_critic_1 = layers.Dense(num_hidden_1, activation=af)(inputs)
        dense_critic_2 = layers.Dense(num_hidden_2, activation=af)(layers.concatenate([actor, dense_critic_1]))
        critic = layers.Dense(1)(dense_critic_2)

        self.model = keras.Model(inputs=inputs, outputs=[actor, critic])

    def call(self, input_state):
        y = self.model(input_state)
        return y

    def display_model_overview(self):
        self.model.summary()
        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=False)

    def save_weights(self, file_loc):
        self.model.save_weights(file_loc)

    def load_weights(self, file_loc):
        self.model.load_weights(file_loc)



class SpgTrainer(keras.models.Model):

    def __init__(self, SPG_network, training_param, buffer):
        super(SpgTrainer, self).__init__()
        self.model = SPG_network
        self.training_param = training_param
        self.buffer = buffer

    def train_step(self):
        with tf.GradientTape() as tape:
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            # Return = SUM_t=0^inf (gamma*reward_t)
            eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
            gamma = self.training_param["gamma"]

            timesteps, states, rewards, chosen_actions = self.buffer.get_experience()

            action_probs, critic_value = self.model(states)

            actions = []
            for t in range(len(timesteps)):
                actions.append(action_probs[t, chosen_actions[t]])
            actions = tf.convert_to_tensor(actions)
            actor_log_prob = tf.math.log(actions)

            returns = []
            discounted_sum = 0
            for r in rewards[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()
            self.buffer.returns = returns

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
                self.buffer.advantage.append(diff)
                actor_losses.append(-actor_log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    self.training_param["loss_func"](tf.expand_dims(crit_value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            self.buffer.losses.append(loss_value)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.buffer.gradients.append(grads)
            self.training_param["optimiser"].apply_gradients(zip(grads, self.model.trainable_variables))
            # Clear the loss and reward history

