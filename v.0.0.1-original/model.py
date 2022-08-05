"""
    Model Module to build neural network and train/test to evaluate individual
"""
from sklearn.utils import shuffle
import tensorflow as tf
from random import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.utils import to_categorical

INPUT_SHAPE = 28

def get_optimizers(function, learning_rate):
    """
        Get optimizer from individual components
    """
    if function == 0:
        opt = optimizers.SGD(learning_rate=learning_rate)
    elif function == 1:
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.1)
    elif function == 2:
        opt = optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    elif function == 3:
        opt = optimizers.Adagrad(learning_rate=learning_rate)
    elif function == 4:
        opt = optimizers.Adamax(learning_rate=learning_rate)
    elif function == 5:
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif function == 6:
        opt = optimizers.Adadelta(learning_rate=learning_rate)
    elif function == 7:
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    return opt

def is_valid_model(conv_sizes, pooling_sizes):
    """
        Check if individual component represent a valid neural network
    """
    shape = INPUT_SHAPE
    for index, value in enumerate(conv_sizes):
        if shape - value + 1 <= 0:
            return False
        shape = shape - value + 1
        if shape - pooling_sizes[index] < 0:
            return False
        shape = (shape - pooling_sizes[index]) // pooling_sizes[index] + 1
    return True

def load_data():
    """
        Load MNIST train and test data then preprocess
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
    x_train, x_test = x_train / 255., x_test / 255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test

class Model():
    """
        Model class to build and train/test neural network from individual components
    """
    def __init__(self, test_mode):
        self.test = test_mode
        self.x_train, self.y_train, self.x_test, self.y_test = load_data()
        self.rgl_dct = {
            0: regularizers.l1(1e-4),
            1: regularizers.l2(1e-4),
            2: regularizers.l1_l2(l1=1e-4, l2=1e-4),
            3: None
        }

    def choose_data(self):
        """
            Choose a random 5000 samples from dataset to shorten trainning time
        """
        small_x, small_y = shuffle(self.x_train, self.y_train)
        return small_x[:5000], small_y[:5000]

    def add_dense_layers(self, components, model):
        """
            Build model's dense layers from individual components
        """
        new_model = model
        if components["nd"] == 1:
            if components["dt"][0] == "feed-forward":
                new_model.add(layers.Flatten())
                new_model.add(
                    layers.Dense(components["dn"][0],
                                 kernel_regularizer=self.rgl_dct[components["dr"][0]],
                                 activation=components["da"][0]))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "recurrent":
                shape = new_model.layers[-1].output_shape
                new_model.add(
                    tf.keras.layers.Reshape(
                        (shape[1] * shape[2], shape[3]),
                        input_shape=shape))
                new_model.add(layers.SimpleRNN(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
        else:
            if components["dt"][0] == "feed-forward" and components["dt"][1] == "feed-forward":
                new_model.add(layers.Flatten())
                new_model.add(layers.Dense(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0]))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Dense(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]],
                    activation=components["da"][1]))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))
            if components["dt"][0] == "feed-forward" and components["dt"][1] == "recurrent":
                new_model.add(layers.Flatten())
                new_model.add(layers.Dense(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] // 2, 2),
                    input_shape=last_shape
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]],
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))
            if components["dt"][0] == "recurrent" and components["dt"][1] == "feed-forward":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]),
                    input_shape=last_shape
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Dense(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]],
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))
            if components["dt"][0] == "recurrent" and components["dt"][1] == "recurrent":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]),
                    input_shape=last_shape
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0],
                    return_sequences=True
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.SimpleRNN(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]],
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))
        new_model.add(layers.Dense(10))
        return new_model

    def build_model(self, components):
        """
            Create full neural network from individual components
        """
        model = models.Sequential()
        model.add(layers.Conv2D(
            components["ck"][0],
            (components["cs"][0], components["cs"][0]),
            activation=components["ca"][0],
            input_shape=(28, 28, 1)
        ))
        model.add(layers.MaxPooling2D((components["cp"][0], components["cp"][0])))

        for i in range(1, components["nc"]):
            model.add(layers.Conv2D(
                components["ck"][i],
                (components["cs"][i], components["cs"][i]),
                activation=components["ca"][i],
            ))
            model.add(layers.MaxPooling2D((components["cp"][i], components["cp"][i])))

        model = self.add_dense_layers(components, model)

        return model

    def evaluate(self, components):
        """
            Get individual fitness by evaluating its model built from components
        """
        if self.test:
            return random()
        if not is_valid_model(components["cs"], components["cp"]):
            return 0
        model = self.build_model(components)
        opt = get_optimizers(components["f"], components["n"])
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        small_x, small_y = self.choose_data()
        batch_size = components["b"]
        model.fit(
            small_x, small_y, epochs=2,
            validation_data=(self.x_test, self.y_test),
            batch_size=batch_size, verbose=0)
        _, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        del model
        return test_acc
