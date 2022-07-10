import numpy as np
import matplotlib.pyplot as plt
import random
import string


def new_char():
    """
    returns a random char
    """
    return random.choice(string.punctuation + string.ascii_letters + string.whitespace)


class Representative:
    """
    class for each representative in the population
    """
    def __init__(self, phrase_length):
        """
        constructor
        phrase_length (int): length of a target phrase
        returns an object of a representative with
        char_array (array): list of chars/phrase
        fitness (float): score of fitness of our representative to the target
        """
        self.phrase_length = phrase_length
        self.char_array = []
        self.fitness = 0.0
        self.create()

    def create(self):
        """
        method for creating an object of a Representative generating random sequence of chars/phrase
        """
        for i in range(self.phrase_length):
            self.char_array.append(new_char())
        return

    def get(self):
        """
        return:
        phrase (str)
        """
        return ''.join(self.char_array)

    def evaluate(self, target):
        """
        arg:
        target (str): target phrase
        method for calculating the fitness of each representative
        """
        assert self.phrase_length == len(target)
        self.fitness = (np.array(list(target)) == np.array(self.char_array)).sum() / len(target)
        return

    def crossover(self, father):
        """
        arg:
        father (Representative): second parent
        return:
        child (Representative): result of crossover of two people
        """
        m = np.floor(random.random() * self.phrase_length)  # select a point
        child = Representative(father.phrase_length)  # initialize a Representative instance

        # crossover of parents depending on the point
        for i in range(father.phrase_length):
            if i < m:
                child.char_array[i] = self.char_array[i]
            else:
                child.char_array[i] = father.char_array[i]
        return child

    def mutate(self, rate):
        """
        arg:
        rate (float): mutation rate
        method for a representative to mutate
        """
        for i in range(len(self.char_array)):
            if random.random() < rate:
                self.char_array[i] = new_char()
        return


class Population:
    """
    population class
    """
    def __init__(self, target, mutation_rate, number, method_one=True):
        self.target = target
        self.mutation_rate = mutation_rate
        self.generations = 0
        self.population = []
        self.mating_pool = []
        self.progress = []  # array for saving the progress of fitness of the best candidate
        self.best_score = 1
        self.method_one = method_one
        self.best = None  # best representative
        self.finished = False  # indicator to finish
        # generate population
        for i in range(number):
            self.population.append(Representative(len(self.target)))
        # calculate fitness for each representative
        self.calculate_fitness()

    def calculate_fitness(self):
        """
        method for calculating fitness
        """
        for p in self.population:
            p.evaluate(self.target)

    def natural_selection(self):
        """
        generates the mating pool for future generations
        """
        self.mating_pool = []

        if self.method_one:
            max_fitness = max([rep.fitness for rep in self.population])
            min_fitness = min([rep.fitness for rep in self.population])
            for rep in self.population:
                current_fitness = int((rep.fitness - min_fitness) * 100 / (max_fitness - min_fitness))
                self.mating_pool.extend([rep] * current_fitness)

        if not self.method_one:
            for rep in self.population:
                current_fitness = int(rep.fitness * len(self.target))
                self.mating_pool.extend([rep] * current_fitness)

    def generate(self):
        """
        method that randomly chooses parents from mating pool and generates children for next population
        """
        for i in range(len(self.population)):
            mother = random.choice(self.mating_pool)
            father = random.choice(self.mating_pool)
            child = mother.crossover(father)
            child.mutate(self.mutation_rate)
            self.population[i] = child
        self.generations += 1

    def get_best(self):
        """
        return:
        representative (Representative): "best" representative
        """
        return self.population[np.argmax([rep.fitness for rep in self.population])]

    def get_avg_fitness(self):
        """
        return:
        avg (float): average fitness score for a population
        """
        avg = np.mean([rep.fitness for rep in self.population])
        return avg

    def evaluate(self):
        """
        finish loop when target is reached
        """
        if self.get_best().fitness == self.best_score:
            self.finished = True
        return


class GARunner:
    def __init__(self, target, mutation_rate, init_population_size, max_iterations=1000):
        self.target = target
        self.mutation_rate = mutation_rate
        self.init_population_size = init_population_size
        self.population = Population(target, mutation_rate, init_population_size)
        self.max_iterations = max_iterations

    def evolve(self):
        self.population.natural_selection()
        self.population.generate()
        self.population.calculate_fitness()
        self.population.evaluate()
        self.population.progress.append(self.population.get_best().fitness)

        print("***")
        print("Population's fitness: ", self.population.get_avg_fitness())
        print("Best fitness: ", self.population.get_best().fitness)
        print("Best match: ", self.population.get_best().get())
        print("Generation: ", self.population.generations)

    def run(self):
        iterations = 0
        while not self.population.finished and iterations < self.max_iterations:
            self.evolve()
            iterations += 1

    def plot(self, figsize=(15, 8)):
        plt.figure(figsize=figsize)
        plt.plot(self.population.progress)
        plt.show()


def main():
    np.random.seed(42)
    target = 'GA is easy peasy lemon squeezy!'
    mutation_rate = 0.01
    init_population_size = 2000
    runner = GARunner(target, mutation_rate, init_population_size)
    runner.run()
    runner.plot()


if __name__ == '__main__':
    main()
