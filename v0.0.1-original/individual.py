"""
    Individual module for each object in the population
"""
from random import randint, random

def binary_to_decimal(bits):
    """
        Convert from binary representation to decimal number
    """
    return int("".join(map(str, bits)), 2)

class Individual():
    """
        Individual class
    """
    def __init__(self, *args):
        parameters = args[0]
        self.gene_length = parameters[0]
        self.activation_dict = parameters[1]
        self.dense_type_dict = parameters[2]
        self.learning_rate_dict = parameters[3]
        if len(args) == 1:
            self.gene = [randint(0, 1) for i in range(self.gene_length)]
        elif len(args) == 2:
            self.gene = args[1]
        else:
            self.gene = args[1]
            self.fitness = args[2]
            self.adjusted_fitness = args[3]

    def evaluate(self, model):
        """
            Evaluate the individual with Model class
            Get components from the individual gene then build the network to train/test
        """
        components = self.get_components()
        self.fitness = model.evaluate(components)

    def mutate(self):
        """
            Bit-flipping mutation
            Each bit in the individual gene has a MUTATION_RATE odd to be changed
        """
        for i in range(self.gene_length):
            odd = random() <= 0.015
            self.gene[i] = int(not self.gene[i]) if odd else self.gene[i]

    def get_batch_size(self):
        """
            Get the batch size value from the individual gene
        """
        return [25, 50, 100, 15][binary_to_decimal(self.gene[:2])]

    def get_convol_layers_num(self):
        """
            Get the number of convolutional layers from the individual gene
        """
        return 1 + binary_to_decimal(self.gene[2:4])

    def get_kernels_num(self, layers_num):
        """
            Get number of kernels in each convolutional layer from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[4 + i * 10: 4 + i * 10 + 3]
            result.append(2 ** (binary_to_decimal(binary) + 1))
        return result

    def get_kernel_sizes(self, layers_num):
        """
            Get the kernel size value from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[7 + i * 10: 7 + i * 10 + 3]
            result.append(2 + binary_to_decimal(binary))
        return result

    def get_pooling(self, layers_num):
        """
            Get the pooling size from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[10 + i * 10: 10 + i * 10 + 3]
            result.append(1 + binary_to_decimal(binary))
        return result

    def get_convol_activation(self, layers_num):
        """
            Get the convolutional layers's activation from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[13 + i * 10: 13 + i * 10 + 1]
            result.append(self.activation_dict[binary_to_decimal(binary)])
        return result

    def get_dense_layers_num(self):
        """
            Get the number of dense layers from the individual gene
        """
        return 1 + binary_to_decimal([self.gene[44]])

    def get_dense_type(self, layers_num):
        """
            Get each dense layers's type from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[45 + i * 8: 45 + i * 8 + 1]
            result.append(self.dense_type_dict[binary_to_decimal(binary)])
        return result

    def get_neurons_num(self, layers_num):
        """
            Get the number of neuron in each layers from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[46 + i * 8: 46 + i * 8 + 3]
            result.append(2 ** (binary_to_decimal(binary) + 3))
        return result

    def get_dense_activation(self, layers_num):
        """
            Get the dense layers's activation from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[49 + i * 8: 49 + i * 8 + 1]
            result.append(self.activation_dict[binary_to_decimal(binary)])
        return result

    def get_regularization(self, layers_num):
        """
            Get the regularization function from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[50 + i * 8: 50 + i * 8 + 2]
            result.append(binary_to_decimal(binary))
        return result

    def get_dropout(self, layers_num):
        """
            Get the dropout probability from the individual gene
        """
        result = []
        for i in range(layers_num):
            binary = self.gene[52 + i * 8: 52 + i * 8 + 1]
            result.append(binary_to_decimal(binary) / 2)
        return result

    def get_optimizer(self):
        """
            Get the optimizer function from the individual gene
        """
        binary = self.gene[61: 64]
        return binary_to_decimal(binary)

    def get_learning_rate(self):
        """
            Get the learning rate value from the individual gene
        """
        binary = self.gene[64: 67]
        return self.learning_rate_dict[binary_to_decimal(binary)]

    def get_components(self):
        """
            Get all components from the individual gene
        """
        dct = {}

        dct["b"] = self.get_batch_size()

        # Convolutional layers
        dct["nc"] = self.get_convol_layers_num()
        dct["ck"] = self.get_kernels_num(dct["nc"])
        dct["cs"] = self.get_kernel_sizes(dct["nc"])
        dct["cp"] = self.get_pooling(dct["nc"])
        dct["ca"] = self.get_convol_activation(dct["nc"])

        # Dense layers
        dct["nd"] = self.get_dense_layers_num()
        dct["dt"] = self.get_dense_type(dct["nd"])
        dct["dn"] = self.get_neurons_num(dct["nd"])
        dct["da"] = self.get_dense_activation(dct["nd"])
        dct["dd"] = self.get_dropout(dct["nd"])
        dct["dr"] = self.get_regularization(dct["nd"])

        # Learning parameters
        dct["n"] = self.get_learning_rate()
        dct["f"] = self.get_optimizer()

        return dct

    def to_string(self):
        """
            Get string representation of the individual's gene
        """
        return "".join(map(str, self.gene))
