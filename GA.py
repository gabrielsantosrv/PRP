from PR_Problem import PRProblem
from utils import read_instance
import random
from random import randrange

class PRP_Genetic:

    def __init__(self, prp: PRProblem):
        self.prp = prp

    def generate_chromosome(self):
        route = []
        for i in range(1, self.prp.n_customers+1):
            route.append(i)
        for i in range(self.prp.n_customers+1, self.prp.n_customers+self.prp.fleet_size):
            route.append(-1)
        random.shuffle(route)
        route.append(-1) #termina a rota com um separador

        speed = [random.randint(self.prp.min_speed, self.prp.max_speed) for _ in range(0, self.prp.n_customers+self.prp.fleet_size)]

        chromosome = (route, speed)
        return chromosome

    def mutate(self, chromosome, prob):
        n = len(chromosome)
        for _ in range(n):
            if random.random() < prob:
                index1 = randrange(0, len(chromosome))
                index2 = randrange(index1, len(chromosome))

                chromosome_mid = chromosome[index1:index2]
                chromosome_mid.reverse()

                chromosome = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
        return chromosome

#OX crossover - crossover para rota
def crossover(parent1, parent2):
    size = len(parent1)
    parent1 = parent1.copy()
    parent2 = parent2.copy()
    parent1_aux = parent1.copy()
    parent2_aux = parent2.copy()

    index1 = 3
    index2 = 6
    child1 = [0]*size
    child2 = [0]*size

    for i in range(index1,index2):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
        parent1_aux.remove(parent2[i])
        parent2_aux.remove(parent1[i])

    for i in range(0, len(parent1_aux)):
        if i < index1:
            child1[i] = parent2_aux[i]
            child2[i] = parent1_aux[i]
        else:
            child1[i+index2-index1] = parent2_aux[i]
            child2[i+index2-index1] = parent1_aux[i]

    # print(parent1)
    # print(parent2)
    # print(parent1_aux)
    # print(parent2_aux)
    # print(child1)
    # print(child2)
    #sortear posições do separator






if __name__ == '__main__':
    # instance_name = "UK10_01"
    # instance = read_instance(inst_name=instance_name)
    # prp = PRP_Genetic(instance)
    # prp.generate_chromosome()
    parent1 = [9, 2, 7, 5, 4, 3, 6, 1, 8]
    parent2 = [2, 8, 3, 6, 9, 5, 7, 4, 1]
    crossover(parent1, parent2)
