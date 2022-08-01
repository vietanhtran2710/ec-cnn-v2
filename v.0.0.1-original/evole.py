from random import randint
from tracker import Tracker
from individual import Individual
from population import Population
from model import Model

POPULATION_SIZE = 50
MAXIMUM_GENERATION = 10
STOP_CONDITION = 10 # Number of generations without improvements
TOURNAMENT_SIZE = 3
MIN_POINTS = 3 # Minium number of points in multipoints crossover
MAX_POINTS = 10 # Maximum number of points in multipoints crossover
MUTATION_RATE = 0.015
ELITE_SIZE = 1
GENE_LENGTH = 67

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

def tournament_selection(population):
    selected, max_fitness = None, 0
    while selected is None:
        for i in range(TOURNAMENT_SIZE):
            contestant = population[randint(0, POPULATION_SIZE - 1)]
            if contestant.adjusted_fitness > max_fitness:
                selected = contestant
                max_fitness = contestant.adjusted_fitness
    return selected

def crossover(parent1, parent2):
    points_num = randint(MIN_POINTS, MAX_POINTS)
    points = [0]
    for i in range(points_num):
        points.append(randint(points[-1] + 1, GENE_LENGTH - (points_num - i)))
    points.append(GENE_LENGTH)
    gene1, gene2 = [], []
    for i in range(len(points) - 1):
        if i % 2 == 0:
            gene1 += parent1.gene[points[i]:points[i + 1]]
            gene2 += parent2.gene[points[i]:points[i + 1]]
        else:
            gene1 += parent2.gene[points[i]:points[i + 1]]
            gene2 += parent1.gene[points[i]:points[i + 1]]
    children1 = Individual(INDIVIDUAL_PARAMETERS, gene1)
    children2 = Individual(INDIVIDUAL_PARAMETERS, gene2)
    return children1, children2

model = Model()

population = Population(INDIVIDUAL_PARAMETERS, POPULATION_SIZE)

# Remove all invalid individual (invalid CNN model structure)
for i in range(POPULATION_SIZE):
    population.populace[i].evaluate()
    while population.populace[i].fitness == 0:
        population.populace[i] = Individual(INDIVIDUAL_PARAMETERS)
        population.populace[i].evaluate()
population.calculate_ajusted_fitness()

tracker = Tracker(STOP_CONDITION)
tracker.update_elitism(population.populace)

print("Generation " + str(tracker.generation_count))
population.print()
tracker.print()

# Population evolution
for i in range(1, MAXIMUM_GENERATION):
    print("Generation", i + 1)
    print("".join(list(map(str, tracker.best_individual.gene))), tracker.best_individual.fitness)

    # Create parent pool for mating by tournament selection
    pool = []
    for j in range(POPULATION_SIZE):
        pool.append(tournament_selection(population.populace))

    # Create offsrping for next generation by crossover then mutate
    next_generation = []
    for j in range(0, POPULATION_SIZE, 2):
        parent1, parent2 = pool[j], pool[j + 1]
        children1, children2 = crossover(parent1, parent2)
        children1.mutate()
        children2.mutate()
        next_generation += [children1, children2]
    for individual in next_generation:
        individual.evaluate()

    # Remove the worst individual
    worst, min_fitness = None, 100
    for i in range(POPULATION_SIZE):
        if next_generation[i].fitness < min_fitness:
            worst, min_fitness = i, next_generation[i].fitness
    del next_generation[worst]

    # Add the best individual from the previous generation
    next_generation.append(tracker.elitism())

    population.populace = next_generation
    population.calculate_ajusted_fitness()
    tracker.update_elitism(population.populace)
    tracker.generation_count += 1
    population.print()

    # Got 10 generations without improvements
    if tracker.stop_condition():
        break

# save_data_to_file(population, tracker)
tracker.print()
