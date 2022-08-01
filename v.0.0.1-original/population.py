from individual import Individual

def count_same_element(array1, array2):
    return [array1[i] == array2[i] for i in range(len(array1))].count(True)

def get_similarity(individual1, individual2):
    nc1 = individual1.get_convol_layers_num()
    nc2 = individual2.get_convol_layers_num()
    nd1 = individual1.get_dense_layers_num()
    nd2 = individual2.get_dense_layers_num()
    if nc1 == nc2 and nd1 == nd2:
        properies_num = nc1 * 4 + nd1 * 5 + 2
        same_count = 0

        f1, f2 = individual1.get_optimizer(), individual2.get_optimizer()
        if f1 == f2:
            same_count += 1
        n1 = individual1.get_learning_rate()
        n2 = individual1.get_learning_rate()
        if n1 == n2:
            same_count += 1

        ck1 = individual1.get_kernels_num(nc1)
        ck2 = individual2.get_kernels_num(nc2)
        same_count += count_same_element(ck1, ck2)
        cs1 = individual1.get_kernel_sizes(nc1)
        cs2 = individual2.get_kernel_sizes(nc2)
        same_count += count_same_element(cs1, cs2)
        cp1 = individual1.get_pooling(nc1)
        cp2 = individual2.get_pooling(nc2)
        same_count += count_same_element(cp1, cp2)
        ca1 = individual1.get_convol_activation(nc1)
        ca2 = individual2.get_convol_activation(nc2)
        same_count += count_same_element(ca1, ca2)

        dt1 = individual1.get_dense_type(nd1)
        dt2 = individual2.get_dense_type(nd2)
        same_count += count_same_element(dt1, dt2)
        dn1 = individual1.get_neurons_num(nd1)
        dn2 = individual2.get_neurons_num(nd2)
        same_count += count_same_element(dn1, dn2)
        da1 = individual1.get_dense_activation(nd1)
        da2 = individual2.get_dense_activation(nd2)
        same_count += count_same_element(da1, da2)
        dr1 = individual1.get_regularization(nd1)
        dr2 = individual2.get_regularization(nd2)
        same_count += count_same_element(dr1, dr2)
        dd1 = individual1.get_dropout(nd1)
        dd2 = individual2.get_dropout(nd2)
        same_count += count_same_element(dd1, dd2)
        return same_count / properies_num
    else:
        return 0

class Population(object):
    def __init__(self, individual_parameters, population_size):
        self.population_size = population_size
        self.populace = [Individual(individual_parameters) for i in range(self.population_size)]

    def calculate_ajusted_fitness(self):
        for i in range(self.population_size):
            similarity_sum = 0
            for j in range(self.population_size):
                if j != i:
                    sim = get_similarity(self.populace[i], self.populace[j])
                    similarity_sum += sim
            similarity = 1 - (similarity_sum / (self.population_size - 1))
            self.populace[i].adjusted_fitness = self.populace[i].fitness
            self.populace[i].adjusted_fitness *= similarity

    def print(self):
        for individual in self.populace:
            print(individual.to_string(), end = " ")
            print(individual.fitness)