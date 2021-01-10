import random
import time
import pandas as pd

from GA import PRP_Genetic, genetic_algorithm_solver
from utils import read_instance

if __name__ == '__main__':
    random.seed(42)


    for instance_size in [100]:
        # for ref_cost in [578.7815538, 643.0906153]:
        for ref_cost in [2175.292804, 2619.790834]:
            results = {'time': [], 'result': [], 'fleet size': []}
            for i in range(1, 150):
                instance_name = "UK{}_01".format(instance_size)
                #config T7
                instance = read_instance(inst_name=instance_name)
                prp = PRP_Genetic(instance)
                ngen = 1000
                final_mutation_rate = 1 / instance_size
                mutation_rate_decay = (1 - final_mutation_rate) / ngen
                start = time.time()
                fitness_scores, best_chromosome, target_time = genetic_algorithm_solver(prp,
                                                                           population_size=100,
                                                                           ngen=ngen,
                                                                           crossover_rate_decay=1,
                                                                           mutation_rate_decay=mutation_rate_decay,
                                                                           route_mutation_rate=1,
                                                                           min_route_mutation_rate=1 / instance_size,
                                                                           speed_mutation_rate=1 / instance_size,
                                                                           elitism_rate=0.15,
                                                                           maintain_diversity=False,
                                                                           selection_method=prp.tournament_selection,
                                                                           patience=1000,
                                                                           time_limit=300,
                                                                           ref_cost=ref_cost
                                                                           )

                n_routes = sum([1 for item in best_chromosome if item[0] == -1])
                print(best_chromosome)
                # plot_metrics(fitness_scores)

                end = time.time()
                min_fit = min(fitness_scores)

                results['result'].append(min_fit)
                results['time'].append(target_time)
                results['fleet size'].append(n_routes)

            df = pd.DataFrame(results)
            df.to_csv('ttplots_uk_{}_configB_{}.csv'.format(instance_size, ref_cost))


