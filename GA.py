import copy
import time

from PR_Problem import PRProblem
from model_constants import *
from utils import read_instance
import random
from random import randrange
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SEPARATOR = -1


class PRP_Genetic:
    def __init__(self, prp: PRProblem):
        self.prp = prp

    def generate_chromosome(self):
        """
        Generate a random chromosome
        :return: chromosome
        """
        route = []
        for i in range(1, self.prp.n_customers + 1):
            customer = i
            speed = random.randint(self.prp.min_speed, self.prp.max_speed)
            route.append([customer, speed])
        random.shuffle(route)
        routes = self.split_route(route)

        chromosome = []
        for route in routes:
            speed_sep = random.randint(self.prp.min_speed, self.prp.max_speed)
            chromosome += route + [[SEPARATOR, speed_sep]]

        return chromosome

    def generate_initial_population(self, population_size):
        """
        Generate a population with ``population_size`` individuals
        :param population_size: number of individuals of the population being generated
        :return: new population
        """
        return [self.generate_chromosome() for _ in range(population_size)]

    def generate_next_population(self, current_population, n_chromosomes,
                                 mutation_rate, speed_mutation_rate, mantain_diversity):
        """

        :param current_population: the current population
        :param n_chromosomes: the number of individuals selected from tournament
        :param k_tournament: the number of individuals involved in the tournament
        :param mutation_rate: route mutation rate, a real number in the interval [0,1]
        :param speed_mutation_rate:  speed mutation rate, a real number in the interval [0,1]
        :return: new population
        """
        winner_chromosomes = self.tournament_selection(current_population, n_chromosomes)
        best_chromosome = sorted(current_population, key=self.fitness)[0]
        winner_chromosomes.append(best_chromosome)

        new_children = []
        qtd_new_children = len(current_population) - n_chromosomes - 1
        random.shuffle(current_population)
        for i in range(0, qtd_new_children, 2):
            parent1 = current_population[i]
            parent2 = current_population[i + 1]
            if mantain_diversity:
                is_new = False
                while not is_new:
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1, mutation_rate=mutation_rate, speed_mutation_rate=speed_mutation_rate)
                    child2 = self.mutate(child2, mutation_rate=mutation_rate, speed_mutation_rate=speed_mutation_rate)
                    child1 = self.add_separator(child1)
                    child2 = self.add_separator(child2)

                    # Take the best among the 4 chromosomes (2 parents and 2 children) involved in the reproduction
                    # new_chromosomes = sorted([child1, child2, parent1, parent2], key=self.fitness)
                    # child1 = new_chromosomes[0]
                    # child2 = new_chromosomes[1]
                    curr_pop = new_children + winner_chromosomes
                    if child1 not in curr_pop and child2 not in curr_pop:
                        is_new = True
            else:
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, mutation_rate=mutation_rate, speed_mutation_rate=speed_mutation_rate)
                child2 = self.mutate(child2, mutation_rate=mutation_rate, speed_mutation_rate=speed_mutation_rate)
                child1 = self.add_separator(child1)
                child2 = self.add_separator(child2)

            new_children.append(child1)
            new_children.append(child2)

        return new_children + winner_chromosomes

    def tournament_selection(self, population, n, k=2):
        """
        Given a population, make k individuals fight against each other and the one with lowest fitness score wins.
        Repeat this process n times.

        :param population: current population
        :param n: the number of individuals selected from the tournament
        :param k: the number of  individuals involved in the tournament
        :return: a list with the winners
        """
        winners = []
        for _ in range(n):
            elements = random.sample(population, k)
            winners.append(min(elements, key=self.fitness))

        return winners

    def fitness(self, chromosome):
        """

        :param chromosome: a chromosome
        :return: its fitness score
        """
        objective = 0
        prev_customer = 0
        const1 = FRICTION_FACTOR * ENGINE_SPEED * ENGINE_DISPLACEMENT * LAMBDA
        const2 = CURB_WEIGHT * GAMMA * LAMBDA * ALPHA
        const3 = GAMMA * LAMBDA * ALPHA
        const4 = BETTA * GAMMA * LAMBDA

        cumulative_time = 0
        cumulative_payload = 0
        total_payloads = self.compute_total_payload(chromosome)
        route_index = 0
        for item in chromosome:
            current_customer = item[0]
            is_separator = current_customer == SEPARATOR
            if is_separator:
                current_customer = 0
            speed = item[1]
            dist = self.prp.dist[prev_customer, current_customer]
            objective += const1 * dist / speed
            objective += const2 * dist
            objective += const4 * dist * (speed ** 2)

            # Driver Cost
            prev_service_time = self.prp.customers[prev_customer]['service_time']
            displacement_time = self.prp.dist[prev_customer, current_customer] / speed
            total_time = cumulative_time + prev_service_time + displacement_time
            ready_time = self.prp.customers[current_customer]['ready_time']
            if total_time < ready_time:
                total_time = ready_time
            cumulative_time = total_time

            if is_separator:
                objective += DRIVER_COST * cumulative_time
                cumulative_time = 0
                cumulative_payload = 0
                prev_customer = 0
                route_index += 1
            else:
                objective += const3 * dist * (total_payloads[route_index] - cumulative_payload)
                cumulative_payload += self.prp.customers[current_customer]['demand']
                prev_customer = current_customer

        return objective

    def split_route(self, chromosome):
        """
        Given a chromosome, compute the routes according to constraints of payload and time.
        :param chromosome: a chromosome without separator
        :return: a list of routes
        """
        not_visited = copy.deepcopy(chromosome)
        routes = []
        max_payload = self.prp.max_payload
        while len(not_visited) > 0:
            not_visited_aux = copy.deepcopy(not_visited)
            current_route = []
            current_payload = 0
            current_time = 0
            prev_customer = 0
            for item in not_visited_aux:
                current_customer = item[0]
                speed = item[1]

                prev_service_time = self.prp.customers[prev_customer]['service_time']
                displacement_time = self.prp.dist[prev_customer, current_customer] / speed
                total_time = current_time + prev_service_time + displacement_time

                ready_time = self.prp.customers[current_customer]['ready_time']
                due_time = self.prp.customers[current_customer]['due_time']
                if total_time < ready_time:
                    total_time = ready_time

                if (current_payload + self.prp.customers[current_customer]['demand'] <= max_payload) and \
                        (ready_time <= total_time <= due_time):
                    current_payload += self.prp.customers[current_customer]['demand']
                    current_time = total_time
                    not_visited.remove(item)
                    current_route.append(item)  # item is a pair [customer, speed]
                    prev_customer = current_customer
            routes.append(current_route)
        return routes

    def mutate(self, chromosome, mutation_rate, speed_mutation_rate):
        """
        Takes a part of the route ``start`` to ``end``, randomly selected, and reverse it.
        Assign random values to speeds
        Ex.

        route: [1,2,|3,4,5,6,|7,8]  =>  [1,2,|6,5,4,3,|7,8]
                    |S     E |               |S     E |

        :param chromosome: array representing the routes and speeds
        :param mutation_rate: a real value in the interval [0, 1]
        :param speed_mutation_rate: a real value in the interval [0, 1]
        :return: mutated chromosome
        """
        _chromosome = copy.deepcopy(chromosome)
        # remove separators
        _chromosome = [x for x in _chromosome if x[0] != SEPARATOR]

        n = len(chromosome)
        is_mutated = False
        for i in range(n):
            if random.random() < speed_mutation_rate:
                _chromosome[i][1] = random.randint(self.prp.min_speed, self.prp.max_speed)
                is_mutated = True

            if random.random() < mutation_rate:
                start = randrange(0, n)
                end = randrange(start, n)

                chromosome_mid = chromosome[start:end]
                chromosome_mid.reverse()
                _chromosome = chromosome[0:start] + chromosome_mid + chromosome[end:]
                is_mutated = True

        return _chromosome

    def add_separator(self, chromosome):
        chromosome = copy.deepcopy(chromosome)
        # TODO: Caso ocorrer uma mutação, atribuimos valores aleatórios para a velocidade do separador
        routes = self.split_route(chromosome)
        chromosome = []
        for route in routes:
            speed_sep = random.randint(self.prp.min_speed, self.prp.max_speed)
            chromosome += route + [[SEPARATOR, speed_sep]]
        return chromosome

    def crossover(self, parent1, parent2):
        """
        OX crossover: ``Given two parents, two random crossover points are selected partitioning them into a left, middle
        and right portion. The ordered two-point crossover behaves in the following way: child1 inherits its left and
        right section from parent1, and its middle section is determined´´.
        (Accessed from https://www.researchgate.net/publication/331580514_Immune_Based_Genetic_Algorithm_to_Solve_a_Combinatorial_Optimization_Problem_Application_to_Traveling_Salesman_Problem_Volume_5_Advanced_Intelligent_Systems_for_Computing_Sciences)

        Ex.
        Parent1: [9,2,7,|5,4,3,|6,1,8]      Parent1: [9,2,7,|5,4,3,|6,1,8]
        Child1:  [2,8,6,|5,4,3,|9,7,1]      Child2:  [2,7,4,|6,9,5,|3,1,8]
        Parent2: [2,8,3,|6,9,5,|7,4,1]      Parent2: [2,8,3,|6,9,5,|7,4,1]

        After OX Crossover, we select randomly positions to separators

        :param parent1: array representing a route
        :param parent2: array representing a route
        :return: the newly generated children
        """
        parent1 = copy.deepcopy(parent1)
        parent2 = copy.deepcopy(parent2)

        # remove separators, turning the VRP representation into TSP representation
        parent1 = [x for x in parent1 if x[0] != SEPARATOR]
        parent2 = [x for x in parent2 if x[0] != SEPARATOR]
        route_size = len(parent1)
        parent1_aux = copy.deepcopy(parent1)
        parent2_aux = copy.deepcopy(parent2)

        n = len(parent1)
        index1 = randrange(0, n)
        index2 = randrange(index1, n)

        child1 = [0] * route_size
        child2 = [0] * route_size

        for i in range(index1, index2):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
            for j in range(0, len(parent1_aux)):
                if parent1_aux[j][0] == parent2[i][0]:
                    parent1_aux.remove(parent1_aux[j])
                    break
            for j in range(0, len(parent2_aux)):
                if parent2_aux[j][0] == parent1[i][0]:
                    parent2_aux.remove(parent2_aux[j])
                    break

        for i in range(0, len(parent1_aux)):
            if i < index1:
                child1[i] = parent2_aux[i]
                child2[i] = parent1_aux[i]
                self.change_speed(child1[i], child2[i])
            else:
                child1[i + index2 - index1] = parent2_aux[i]
                child2[i + index2 - index1] = parent1_aux[i]
                self.change_speed(child1[i + index2 - index1], child2[i + index2 - index1])

        return child1, child2

    def change_speed(self, gene1, gene2):
        """
        Given two genes, change their speeds with probability 0.5
        :param gene1:
        :param gene2:
        :return:
        """
        if random.random() < 0.5:
            aux = gene1[1]
            gene1[1] = gene2[1]
            gene2[1] = aux

    def compute_total_payload(self, chromosome):
        """
        Given a chromosome, compute the total payload of each route
        :param chromosome: a chromosome
        :return: a list with the total payload of each route
        """
        total_payloads = []
        current_payload = 0
        for item in chromosome:
            customer = item[0]
            if customer == SEPARATOR:
                total_payloads.append(current_payload)
                current_payload = 0
            else:
                current_payload += self.prp.customers[customer]['demand']

        return total_payloads


def genetic_algorithm_solver(instance, population_size, ngen, crossover_rate_decay,
                             mutation_rate_decay, speed_mutation_rate, time_limit=600,
                             maintain_diversity=False):
    prp = PRP_Genetic(instance)
    population = prp.generate_initial_population(population_size)

    start = time.time()
    fitness_scores = []
    mutation_rate = 1
    crossover_rate = 1
    for n in range(0, ngen):
        n_parents = round(population_size * (1 - crossover_rate))
        n_parents = (n_parents if n_parents % 2 == 0 else n_parents - 1)
        population = prp.generate_next_population(population, n_parents, mutation_rate,
                                                  speed_mutation_rate, maintain_diversity)

        best_chromosome = min(population, key=prp.fitness)

        fitness_score = prp.fitness(best_chromosome)
        print("Generation", n, "Best Fitness", fitness_score)
        fitness_scores.append(fitness_score)
        mutation_rate -= mutation_rate_decay
        crossover_rate *= crossover_rate_decay
        end = time.time()
        if (end - start) >= time_limit:
            break

    return fitness_scores, best_chromosome


def plot_metrics(scores, filepath=None):
    formatted_dict = {'Generations': [],
                      'Fitness Scores': [], }

    for i, score in enumerate(scores):
        formatted_dict['Generations'].append(i)
        formatted_dict['Fitness Scores'].append(score)

    df_metrics = pd.DataFrame(formatted_dict)
    sns.lineplot(data=df_metrics, x='Generations', y='Fitness Scores')
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)


if __name__ == '__main__':
    random.seed(42)
    results = {'time': [], 'result': [], 'fleet size': []}

    for i in range(1, 2):
        if i < 10:
            instance_name = "{}{}".format("UK10_0", i)
        else:
            instance_name = "{}{}".format("UK10_", i)

        instance = read_instance(inst_name=instance_name)
        start = time.time()
        fitness_scores, best_chromosome = genetic_algorithm_solver(instance, population_size=100, ngen=1000,
                                                                   crossover_rate_decay=0.99, mutation_rate_decay=0.85/1000,
                                                                   speed_mutation_rate=0.25, maintain_diversity=True)
        n_routes = sum([1 for item in best_chromosome if item[0] == -1])
        plot_metrics(fitness_scores)
    #     end = time.time()
    #     min_fit = min(fitness_scores)
    #
    #     results['result'].append(min_fit)
    #     results['time'].append((end-start))
    #     results['fleet size'].append(n_routes)
    #
    # df = pd.DataFrame(results)
    # df.to_csv('ga_results.csv')
    # plot_metrics(fitness_scores)
