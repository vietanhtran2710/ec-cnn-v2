"""
    Evolutionary Computing Module - Modified Original Approach
    This module will be used to run on Google Colab
"""
from random import randint
import sys
from tracker import Tracker
from individual import Individual
from population import Population
from model import Model

POPULATION_SIZE = 25
MAXIMUM_GENERATION = 30
STOP_CONDITION = 30 # Number of generations without improvements
TOURNAMENT_SIZE = 3
MIN_POINTS = 3 # Minium number of points in multipoints crossover
MAX_POINTS = 10 # Maximum number of points in multipoints crossover
MUTATION_RATE = 0.015
ELITE_SIZE = 1
GENE_LENGTH = 67
TEST_MODE = sys.argv[1] == "-t"

LEARNING_RATE_DICT = {
    0: 1 * 10 ** (-5), 1: 5 * 10 ** (-5),
    2: 1 * 10 ** (-4), 3: 5 * 10 ** (-4),
    4: 1 * 10 ** (-3), 5: 5 * 10 ** (-3),
    6: 1 * 10 ** (-2), 7: 5 * 10 ** (-2),
}

DENSE_TYPE_DICT = {
    0: "recurrent", 1: "feed-forward"
}

REGULARIZATION_DICT = {
    0: "l1", 1: "l2", 2: "l1l2", 3: None
}

ACTIVATION_DICT = {
    0: "relu",
    1: "linear"
}

INDIVIDUAL_PARAMETERS = [GENE_LENGTH, ACTIVATION_DICT, DENSE_TYPE_DICT, LEARNING_RATE_DICT]

def tournament_selection(current_population):
    """
        Tournament Selection for Parent Pool Selecting Phase
        Choose 3 random individual from the population with replacement
        The individual with highest adjusted fitness win the tournament, get added to the pool
    """
    selected, max_fitness = None, 0
    while selected is None:
        for _i in range(TOURNAMENT_SIZE):
            contestant = current_population[randint(0, POPULATION_SIZE - 1)]
            if contestant.adjusted_fitness > max_fitness:
                selected = contestant
                max_fitness = contestant.adjusted_fitness
    return selected

def crossover(_parent1, _parent2):
    """
        Multipoint crossover for Crossover Phase
        Split both parent gene into multiple segments
        Swap segments at odd index, offsprings genes are achieved
    """
    points_num = randint(MIN_POINTS, MAX_POINTS)
    points = [0]
    for _i in range(points_num):
        points.append(randint(points[-1] + 1, GENE_LENGTH - (points_num - _i)))
    points.append(GENE_LENGTH)
    gene1, gene2 = [], []
    for _i in range(len(points) - 1):
        if _i % 2 == 0:
            gene1 += _parent1.gene[points[_i]:points[_i + 1]]
            gene2 += _parent2.gene[points[_i]:points[_i + 1]]
        else:
            gene1 += _parent2.gene[points[_i]:points[_i + 1]]
            gene2 += _parent1.gene[points[_i]:points[_i + 1]]
    _children1 = Individual(INDIVIDUAL_PARAMETERS, gene1)
    _children2 = Individual(INDIVIDUAL_PARAMETERS, gene2)
    return _children1, _children2

MODEL = Model(TEST_MODE)

POPULATION = Population(INDIVIDUAL_PARAMETERS, POPULATION_SIZE)

# Remove all invalid individual (invalid CNN model structure)
for _i in range(POPULATION_SIZE):
    POPULATION.populace[_i].evaluate(MODEL)
    while POPULATION.populace[_i].fitness == 0:
        POPULATION.populace[_i] = Individual(INDIVIDUAL_PARAMETERS)
        POPULATION.populace[_i].evaluate(MODEL)
POPULATION.calculate_ajusted_fitness()

TRACKER = Tracker(STOP_CONDITION)
TRACKER.update_elitism(POPULATION.populace)

print("Generation " + str(TRACKER.generation_count))
POPULATION.print()
TRACKER.print()

# Population evolution
for _i in range(1, MAXIMUM_GENERATION):
    print("Generation", _i + 1)
    print("".join(list(map(str, TRACKER.best_individual.gene))), TRACKER.best_individual.fitness)

    # Create parent pool for mating by tournament selection
    pool = []
    for j in range(POPULATION_SIZE):
        pool.append(tournament_selection(POPULATION.populace))

    # Create offsrping for next generation by crossover then mutate
    next_generation = []
    for j in range(0, POPULATION_SIZE, 2):
        parent1, parent2 = pool[j], pool[j + 1]
        children1, children2 = crossover(parent1, parent2)
        children1.mutate()
        children2.mutate()
        next_generation += [children1, children2]
    for individual in next_generation:
        individual.evaluate(MODEL)

    POPULATION.populace += next_generation
    POPULATION.populace.sort(key = lambda x: x.fitness, reverse=True)
    POPULATION.populace = POPULATION.populace[:25]
    POPULATION.calculate_ajusted_fitness()
    TRACKER.update_elitism(POPULATION.populace)
    TRACKER.generation_count += 1
    POPULATION.print()

    # Got 10 generations without improvements
    if TRACKER.stop_condition():
        break

# save_data_to_file(population, tracker)
TRACKER.print()
