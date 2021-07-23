#!/usr/bin/env python

import random

import numpy

from deap import algorithms
from deap import base
from deap import cma
from deap import benchmarks
from deap import creator
from deap import tools

IND_SIZE = 10
MIN_VALUE = -5.12
MAX_VALUE = 5.12
SIGMA_INIT = 0.02886751345
NUM_GENS = 500
C_VALUE = 0.87


class StrategyOnePlusOne(cma.StrategyOnePlusLambda):
    ONE_FIFTH = 1.0 / 5.0

    def __init__(self, parent, sigma, c_value, **kwargs):
        super().__init__(parent, sigma, **kwargs)
        self.lambda_ = 1
        self.alpha = (c_value * c_value)

    def update(self, population):
        super().update(population)

        print(self.psucc)
        if self.psucc < self.ONE_FIFTH:
            self.sigma *= self.alpha
        elif self.psucc > self.ONE_FIFTH:
            self.sigma /= self.alpha


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.sphere)


def gen_ind(min_val: float, max_val: float, size: int) -> creator.Individual:
    return creator.Individual(
        (numpy.random.uniform(min_val, max_val) for _ in range(size)))


def main():
    numpy.random.seed()

    parent = gen_ind(MIN_VALUE, MAX_VALUE, IND_SIZE)
    parent.fitness.values = toolbox.evaluate(parent)

    strategy = StrategyOnePlusOne(parent, SIGMA_INIT, C_VALUE)
    toolbox.register("generate", strategy.generate,
                     ind_init=creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaGenerateUpdate(
        toolbox, ngen=NUM_GENS, halloffame=hof, stats=stats)
    # return pop, logbook, hof


if __name__ == "__main__":
    main()
