"""
    Tracker module containing Tracker class
"""
class Tracker():
    """
        Tracker class observing and storing population historical information
    """
    def __init__(self, stop_condition):
        self.generation_count = 1
        self.best_fitness = []
        self.population_history = []
        self.best_individual = None
        self.stop_limit = stop_condition

    def stop_condition(self):
        """
            Check if the stop condition is met to halt evolve process
        """
        if self.generation_count >= self.stop_limit:
            if self.best_fitness[-1] == self.best_fitness[-10]:
                return True
        return False

    def elitism(self):
        """
            Get the strongest individual
        """
        return self.best_individual

    def update_elitism(self, population):
        """
            Update population historical data
        """
        generation = []
        for individual in population:
            generation.append(
                " ".join(
                    [
                        "".join(map(str, individual.gene)),
                        str(individual.fitness),
                        str(individual.adjusted_fitness)
                    ]
                ) + "\n")
        self.population_history.append(generation)
        best, _max = None, 0
        for individual in population:
            if individual.fitness > _max:
                best, _max = individual, individual.fitness
        self.best_individual = best
        self.best_fitness.append(_max)

    def print(self):
        """
            Print tracker data
        """
        print(self.best_fitness)
        print(
            self.best_individual.gene,
            self.best_individual.fitness,
            self.best_individual.adjusted_fitness)
