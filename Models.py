import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.backend.set_floatx('float64')


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

        self.model = keras.Model(inputs=inputs, outputs=[actor, critic], name="SPG_model")

    def call(self, input_state):
        y = self.model(input_state)
        return y

    def display_model_overview(self):
        self.model.summary()
        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)

    def save_weights(self, file_loc):
        self.model.save_weights(file_loc)

    def load_weights(self, file_loc):
        self.model.load_weights(file_loc)

