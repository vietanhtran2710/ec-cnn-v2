"""
    Population module for Evolutionary Computing
"""
from individual import Individual

def count_same_element(array1, array2):
    """
        Get how many components both individual have the same value
    """
    return [array1[i] == array2[i] for i in range(len(array1))].count(True)

def get_similarity(individual1, individual2):
    """
        Get the similarity between two individuals
    """
    nc1 = individual1.get_convol_layers_num()
    nc2 = individual2.get_convol_layers_num()
    nd1 = individual1.get_dense_layers_num()
    nd2 = individual2.get_dense_layers_num()
    if nc1 == nc2 and nd1 == nd2:
        properies_num = nc1 * 4 + nd1 * 5 + 2
        same_count = 0

        function1, function2 = individual1.get_optimizer(), individual2.get_optimizer()
        if function1 == function2:
            same_count += 1
        num1 = individual1.get_learning_rate()
        num2 = individual1.get_learning_rate()
        if num1 == num2:
            same_count += 1

        same_count += count_same_element(
            individual1.get_kernels_num(nc1),
            individual2.get_kernels_num(nc2))
        same_count += count_same_element(
            individual1.get_kernel_sizes(nc1),
            individual2.get_kernel_sizes(nc2))
        same_count += count_same_element(
            individual1.get_pooling(nc1),
            individual2.get_pooling(nc2))
        same_count += count_same_element(
            individual1.get_convol_activation(nc1),
            individual2.get_convol_activation(nc2))
        same_count += count_same_element(
            individual1.get_dense_type(nd1),
            individual2.get_dense_type(nd2))
        same_count += count_same_element(
            individual1.get_neurons_num(nd1),
            individual2.get_neurons_num(nd2))
        same_count += count_same_element(
            individual1.get_dense_activation(nd1),
            individual2.get_dense_activation(nd2))
        same_count += count_same_element(
            individual1.get_regularization(nd1),
            individual2.get_regularization(nd2))
        same_count += count_same_element(
            individual1.get_dropout(nd1),
            individual2.get_dropout(nd2))
        return same_count / properies_num
    return 0

class Population():
    """
        Population class
        Containing multiple Individual instances
    """
    def __init__(self, individual_parameters, population_size):
        self.population_size = population_size
        self.populace = [Individual(individual_parameters) for i in range(self.population_size)]

    def calculate_ajusted_fitness(self):
        """
            Calculate adjusted fitness from main fitness
            The higher similarity with other individuals, the lower adjusted fitness
        """
        for i in range(len(self.populace)):
            similarity_sum = 0
            for j in range(len(self.populace)):
                if j != i:
                    sim = get_similarity(self.populace[i], self.populace[j])
                    similarity_sum += sim
            similarity = 1 - (similarity_sum / (self.population_size - 1))
            self.populace[i].adjusted_fitness = self.populace[i].fitness
            self.populace[i].adjusted_fitness *= similarity

    def slow_remove(self):
        while len(self.populace) > self.population_size:
            self.calculate_ajusted_fitness()
            self.populace.sort(key = lambda x: x.adjusted_fitness, reverse=True)
            self.populace.pop()

    def remove_duplication(self):
        for i in range(len(self.populace)):
            j = i + 1
            while j < len(self.populace):
                while j < len(self.populace) and len(self.populace) > 25 and get_similarity(self.populace[i], self.populace[j]) == 1:
                    del self.populace[j]
                j += 1

    def print(self):
        """
            Print population into
        """
        for individual in self.populace:
            print(individual.to_string(), end=" ")
            print(individual.fitness)
