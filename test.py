import random
import numpy as np
from deap import base, creator, tools, algorithms

def ga_demo(x):
    x = list(x)
    f2 = ((x[0] - 5) / 1.2) ** 2 + (x[1] - 6) ** 2 < 16
    f1 = 6.452 * (x[0] + 0.125 * x[1]) * (np.cos(x[0]) - np.cos(2 * x[1])) ** 2
    f1 = f1 / np.sqrt(0.8 + (x[0] - 4.2) ** 2 + 2 * (x[1] - 7) ** 2)
    f1 = f1 + 3.226 * x[1]
    f = 100 - f1 * f2
    return -100 * (100 - f) / 100

# 创建适应度最小化的问题类
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 定义问题的参数
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)  # 可调整变量的范围
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", ga_demo)  # 评估函数

# 设置遗传算法的参数
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # 变异操作
toolbox.register("select", tools.selTournament, tournsize=3)  # 选择操作

# 创建初始种群
population = toolbox.population(n=50)

# 迭代次数
ngen = 100

# 迭代运行遗传算法
algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=ngen, stats=None, halloffame=None, verbose=True)

# 输出最优解
best_individual = tools.selBest(population, k=1)[0]
best_fitness = best_individual.fitness.values[0]
best_x = best_individual

print("最小值:", best_fitness)
print("最优解:", best_x)
