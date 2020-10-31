
from CartPole import CartPoleEnv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import random

def createModel(num_hidden, num_inputs, num_actions):
    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    actor = layers.Dense(num_actions, activation="softmax")(common)  # y=E(0,1)
    critic = layers.Dense(1)(common)

    model = keras.Model(inputs=inputs, outputs=[actor, critic])
    model.summary()  # Display what overview of model
    return model

def trainModel(max_steps_per_episode, gamma, final_return, optimizer, loss_function):
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0
    # Run until all episodes completed (reward level reached)
    while True:
        # Set environment with random state array: X=(x,xdot,theta,theta_dot)
        state = env.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(state)
                # actor calculates the probabilities of outputs from the given state
                # critic evaluates the action by computing the value function
                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution
                # squeeze removes single-dimensional entries from array shape
                # choose between 0 and 1 with the probabilities calculated from network
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))

                action_probs_history.append(tf.math.log(action_probs[0, action]))
                # log probabilities have better accuracy (better represent small probabilities)

                # Apply the sampled action in our environment
                state, reward, done, info = env.step(action)
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            # Return = SUM_t=0^inf (gamma*reward_t)
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for actor_log_prob, crit_value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up receiving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - crit_value
                actor_losses.append(-actor_log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    loss_function(tf.expand_dims(crit_value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        # Log details
        episode_count += 1
        if episode_count % 10 == 0:
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))

        if running_reward >= final_return:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break
    return model

def runSimulation(simulated_timesteps):
    state = env.reset()
    for _ in range(simulated_timesteps):
        env.render()
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        # action to take based on model:
        action_probabilities, critic_values = model(state)
        action = np.random.choice(num_actions, p=np.squeeze(action_probabilities))
        # get observation from environment based on action taken:
        observation, reward, done, info = env.step(action)
        state = observation
        if done:
            env.close()
            break


if __name__ == "__main__":
    # Configuration parameters for the whole setup
    seed = 42
    gamma_discount = 0.99  # Discount factor for past rewards
    max_episode_steps = 10000  # Max time steps per run
    final_return = 200  # Total return when model completed
    env = CartPoleEnv()  # Create the environment
    env.seed(seed)

    num_inputs = 4  # (x,x_dot,theta,theta_dot)
    num_actions = 2  # (0,1)-(left,right)
    num_hidden = 250  # number of hidden nodes in perceptron
    model = createModel(num_hidden, num_inputs, num_actions)

    optimizer_adam = keras.optimizers.Adam(learning_rate=0.02)
    optimizer_RMSprop = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.01)
    loss_function = keras.losses.Huber()

    #finalModel = trainModel(max_episode_steps,gamma_discount,final_return,optimizer_adam,loss_function)
    #finalModel.save_weights("./model/model_weights")
    model.load_weights("./model/model_weights")

    num_timesteps = 1000
    runSimulation(num_timesteps)

