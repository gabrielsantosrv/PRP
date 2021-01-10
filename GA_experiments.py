import random
import time
import pandas as pd

from GA import PRP_Genetic, genetic_algorithm_solver
from utils import read_instance

def sus_A(elitism, k):
    results = {'time': [], 'result': [], 'fleet size': []}
    for i in range(1, 21):
        if i < 10:
            instance_name = "{}{}".format("UK{}_0".format(instance_size), i)
        else:
            instance_name = "{}{}".format("UK{}_".format(instance_size), i)

        instance = read_instance(inst_name=instance_name)
        prp = PRP_Genetic(instance)
        ngen = 100000
        start = time.time()
        fitness_scores, best_chromosome = genetic_algorithm_solver(prp,
                                                                   population_size=100,
                                                                   ngen=ngen,
                                                                   crossover_rate_decay=1,
                                                                   mutation_rate_decay=0,
                                                                   route_mutation_rate=1 / instance_size,
                                                                   min_route_mutation_rate=1 / instance_size,
                                                                   speed_mutation_rate=1 / instance_size,
                                                                   elitism_rate=elitism,
                                                                   maintain_diversity=False,
                                                                   selection_method=lambda pop, n: prp.sus(pop, n, k=k))
        n_routes = sum([1 for item in best_chromosome if item[0] == -1])
        print(best_chromosome)
        # plot_metrics(fitness_scores)

        end = time.time()
        min_fit = min(fitness_scores)

        results['result'].append(min_fit)
        results['time'].append((end - start))
        results['fleet size'].append(n_routes)

    return results

def sus_B(elitism, k):
    results = {'time': [], 'result': [], 'fleet size': []}
    for i in range(1, 21):
        if i < 10:
            instance_name = "{}{}".format("UK{}_0".format(instance_size), i)
        else:
            instance_name = "{}{}".format("UK{}_".format(instance_size), i)

        instance = read_instance(inst_name=instance_name)
        prp = PRP_Genetic(instance)
        ngen = 100000
        final_mutation_rate = 1 / instance_size
        mutation_rate_decay = (1 - final_mutation_rate) / ngen
        start = time.time()
        fitness_scores, best_chromosome = genetic_algorithm_solver(prp,
                                                                   population_size=100,
                                                                   ngen=ngen,
                                                                   crossover_rate_decay=1,
                                                                   mutation_rate_decay=mutation_rate_decay,
                                                                   route_mutation_rate=1,
                                                                   min_route_mutation_rate=1 / instance_size,
                                                                   speed_mutation_rate=1 / instance_size,
                                                                   elitism_rate=elitism,
                                                                   maintain_diversity=False,
                                                                   selection_method=lambda pop, n: prp.sus(pop, n, k=k))
        n_routes = sum([1 for item in best_chromosome if item[0] == -1])
        print(best_chromosome)
        # plot_metrics(fitness_scores)

        end = time.time()
        min_fit = min(fitness_scores)

        results['result'].append(min_fit)
        results['time'].append((end - start))
        results['fleet size'].append(n_routes)

    return results


def tournament_A(elitism):
    results = {'time': [], 'result': [], 'fleet size': []}
    for i in range(1, 21):
        if i < 10:
            instance_name = "{}{}".format("UK{}_0".format(instance_size), i)
        else:
            instance_name = "{}{}".format("UK{}_".format(instance_size), i)

        instance = read_instance(inst_name=instance_name)
        prp = PRP_Genetic(instance)
        ngen = 100000
        start = time.time()
        fitness_scores, best_chromosome = genetic_algorithm_solver(prp,
                                                                   population_size=100,
                                                                   ngen=ngen,
                                                                   crossover_rate_decay=1,
                                                                   mutation_rate_decay=0,
                                                                   route_mutation_rate=1 / instance_size,
                                                                   min_route_mutation_rate=1 / instance_size,
                                                                   speed_mutation_rate=1 / instance_size,
                                                                   elitism_rate=elitism,
                                                                   maintain_diversity=False,
                                                                   selection_method=prp.tournament_selection)
        n_routes = sum([1 for item in best_chromosome if item[0] == -1])
        print(best_chromosome)
        # plot_metrics(fitness_scores)

        end = time.time()
        min_fit = min(fitness_scores)

        results['result'].append(min_fit)
        results['time'].append((end - start))
        results['fleet size'].append(n_routes)

    return results


def tournament_B(elitism):
    results = {'time': [], 'result': [], 'fleet size': []}
    for i in range(1, 21):
        if i < 10:
            instance_name = "{}{}".format("UK{}_0".format(instance_size), i)
        else:
            instance_name = "{}{}".format("UK{}_".format(instance_size), i)

        instance = read_instance(inst_name=instance_name)
        prp = PRP_Genetic(instance)
        ngen = 100000
        final_mutation_rate = 1 / instance_size
        mutation_rate_decay = (1 - final_mutation_rate) / ngen
        start = time.time()
        fitness_scores, best_chromosome = genetic_algorithm_solver(prp,
                                                                   population_size=100,
                                                                   ngen=ngen,
                                                                   crossover_rate_decay=1,
                                                                   mutation_rate_decay=mutation_rate_decay,
                                                                   route_mutation_rate=1,
                                                                   min_route_mutation_rate=1 / instance_size,
                                                                   speed_mutation_rate=1 / instance_size,
                                                                   elitism_rate=elitism,
                                                                   maintain_diversity=False,
                                                                   selection_method=prp.tournament_selection)
        n_routes = sum([1 for item in best_chromosome if item[0] == -1])
        print(best_chromosome)

        end = time.time()
        min_fit = min(fitness_scores)

        results['result'].append(min_fit)
        results['time'].append((end - start))
        results['fleet size'].append(n_routes)

    return results


if __name__ == '__main__':
    random.seed(42)

    for elitism in [0.15]:
        for k in [5, 50]:
            for instance_size in [10, 20, 100, 200]:
                print("==================== Instance {}, elitism {}, k {}".format(instance_size, elitism, k))
                results = sus_A(elitism, k)
                df = pd.DataFrame(results)
                df.to_csv('ga_{}_config_A_{}_SUS_{}_results.csv'.format(instance_size, elitism, k))

                results = sus_B(elitism, k)
                df = pd.DataFrame(results)
                df.to_csv('ga_{}_config_B_{}_SUS_{}_results.csv'.format(instance_size, elitism, k))

