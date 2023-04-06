import math
import random

import numpy as np
import pandas as pd

POPULATION_SIZE = 100  # 하나의 세대 마다 염색체 개수
MUTATION_RATE = 0.01
SIZE = 100  # 하나의 염색체에서 유전자의 개수


class Chromosome:
    def __init__(self, g=None):
        if g is None:
            g = []
        self.genes = g.copy()  # 유전자는 리스트로 구현된다.
        self.fitness = 0  # 적합도
        if self.genes.__len__() == 0:  # 염색체가 초기 상태이면 초기화한다.
            i = 0
            self.genes.append(0)
            while i < SIZE - 1:  # 서울은 제외
                num = random.randint(1, SIZE - 1)
                while num in self.genes:  # 겹치는 도시가 있으면 다시 뽑는다.
                    num = random.randint(1, SIZE - 1)
                self.genes.append(num)
                i += 1

    def cal_fitness(self):  # 적합도 계산 함수
        self.fitness = 0
        d = DISTANCE[0][self.genes[1]]  # 서울~첫 도시 거리를 더한다.
        for idx in range(1, SIZE - 1):
            d += DISTANCE[self.genes[idx]][self.genes[idx + 1]]  # 서울을 제외한 각 도시 사이의 거리를 더한다.
        d += DISTANCE[self.genes[-1]][0]  # 마지막 도시~서울 거리를 더한다.
        self.fitness = (1 / d)  # 거리가 짧을수록 적합도가 높아야 한다.
        return self.fitness

    def __str__(self):
        return self.genes.__str__()


# 염색체와 적합도를 출력한다.
def print_p(pop):
    i = 0
    for x in pop:
        # print("염색체 #", i, "=", x, "적합도=", x.cal_fitness())
        print("현재 거리 : ", 1 / x.cal_fitness())
        i += 1
    print("")


import itertools


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def total_distance(points):
    return sum(distance(point, points[index + 1]) for index, point in enumerate(points[:-1])) + distance(points[-1], points[0])


def select(population):
    # 일부 도시 쌍에 대한 거리 계산
    distances = {}
    for _ in range(10):  # 랜덤으로 10개의 도시 쌍 선택
        i, j = random.sample(range(SIZE), 2)
        print("select 실행 ")
        if i < j:
            distances[(i, j)] = distance(population[i], population[j])
        else:
            distances[(j, i)] = distance(population[j], population[i])
    # 총 거리가 가장 짧은 2개의 도시 쌍 선택
    shortest_distances = sorted(distances.items(), key=lambda x: x[1])[:2]
    city_pairs = [(population[i], population[j]) for (i, j), _ in shortest_distances]
    return [tuple(pair) for pair in itertools.chain.from_iterable(city_pairs)]


def crossover(population):
    father1, father2 = select(population), select(population)
    mother1, mother2 = select(population), select(population)
    index = random.randint(1, SIZE - 2)  # 교차 지점 선택
    # 첫번째 자식 생성
    temp = list(father1)  # father1 염색체를 복사해 교차로 받아온 유전자를 삭제한다.
    for i in mother1[index:]:
        if i not in temp:
            temp.append(i)
    child1 = tuple(temp)
    # 두번째 자식 생성
    temp = list(mother2)  # mother2 염색체를 복사해 교차로 받아온 유전자를 삭제한다.
    for i in father2[index:]:
        if i not in temp:
            temp.append(i)
    child2 = tuple(temp)
    return (child1, child2)


# 상호 교환 돌연변이 연산
def mutate(c):
    for i in range(1, SIZE):
        if random.random() < MUTATION_RATE:
            g = random.randint(1, SIZE - 1)  # 서울을 제외한 도시 하나를 선택한다.
            while g == i:
                g = random.randint(1, SIZE - 1)
            c.genes[i], c.genes[g] = c.genes[g], c.genes[i]  # 두 유전자를 상호 교환한다.


def read_csv(file):  # csv 파일 읽기, csv file read
    data = pd.read_csv(filename, header=None, names=['x', 'y'])
    cities = np.array(data[['x', 'y']])
    return cities


def calculate_distances(points):
    n = len(points)
    distance_list = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5
            distance_list[i][j] = distance
            distance_list[j][i] = distance
    return distance_list


# 메인 프로그램
filename = '2023_AI_TSP.csv'
temp = read_csv(filename)
DISTANCE = calculate_distances(temp)
population = []
i = 0
# 초기 염색체를 생성하여 객체 집단에 추가한다.
while i < 1000:
    population.append(Chromosome())
    i += 1
count = 0
population.sort(key=lambda x: x.cal_fitness(), reverse=True)
print("세대 번호=", count)
print_p(population)
count = 1
while population[0].cal_fitness():  # sorting을 하기 때문에 [0]을 비교한다.
    new_pop = []
    # 선택과 교차 연산
    for _ in range(POPULATION_SIZE // 2):
        c1, c2 = crossover(population)
        new_pop.append(Chromosome(c1))
        new_pop.append(Chromosome(c2))
    # 자식 세대가 부모 세대를 대체한다.
    # 깊은 복사를 수행한다.
    min_fitness = population[0].cal_fitness()
    population = new_pop.copy();
    # 돌연변이 연산
    for c in population: mutate(c)
    # 출력을 위한 정렬
    population.sort(key=lambda x: x.cal_fitness(), reverse=True)
    print("세대 번호=", count)
    print_p(population)
    count += 1
    # if count ! : break
