import numpy as np
import random


def generate_value(total_item):
    return np.random.randint(1, 100, total_item)


def generate_weight(total_item):
    return np.random.randint(1, 100, total_item)


def is_feasible(chromosome, weight, capacity):
    total = 0
    for i in range(len(chromosome)):
        total += chromosome[i] * weight[i]
    return total <= capacity


def repair_chromosome(chromosome, weight, capacity):
    total_weight = 0
    for i in range(len(chromosome)):
        total_weight += chromosome[i] * weight[i]

    item = sorted(zip(weight, [x for x in range(len(weight))]))
    temp = chromosome
    now = 0
    while total_weight > capacity:
        temp[item[now][1]] = 0
        total_weight -= item[now][0]
        now = now + 1
    return temp


def fitness_value(chromosome, value):
    total = 0
    for i in range(len(chromosome)):
        total += chromosome[i] * value[i]
    return total


def repair_offspring(offspring, weight, capacity):
    new_offspring = []
    for x in offspring:
        if is_feasible(x, weight, capacity):
            new_offspring.append(x)
        else:
            new_offspring.append(repair_chromosome(x, weight, capacity))
    return new_offspring


def hamming_distance(a, b):
    distance = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            distance += 1
    return distance


def active_genes(a):
    return np.sum(a)


def separate_by_gender(population, generation):
    gender = 0
    generation = generation % 2
    female = []
    male = []
    for x in population:
        if (gender % 2) != generation:
            female.append(x)
        else:
            male.append(x)
    return male, female


def female_tournament_selection(female, value, tournament_round):
    tournament_round -= 1
    female_chro = female[np.random.randint(0, len(value))]
    fitness_value_female_chro = fitness_value(female_chro, value)
    for i in range(tournament_round):
        chromosome = female[np.random.randint(0, len(value))]
        fitness_value_temp = fitness_value(chromosome, value)
        if fitness_value_temp > fitness_value_female_chro:
            female_chro = chromosome
            fitness_value_female_chro = fitness_value_temp

    return female_chro


def male_selection(male, female_chro, value, size=-1):
    if size == -1:
        size = len(male) // 2
    male_temp = random.sample(male, size)
    male_hamming_distance = [hamming_distance(x, female_chro) for x in male_temp]
    male_fitness_value = [fitness_value(x, value) for x in male_temp]
    male_active_genes = [active_genes(x) for x in male_temp]
    male_index = [x for x in range(len(male_temp))]
    male_chro_index = max(zip(male_hamming_distance, male_fitness_value, male_active_genes, male_index))[3]
    male_chro = male_temp[male_chro_index]
    return male_chro


def calculate_t(population, value):
    size = len(population)
    all_fitness_value = [fitness_value(x, value) for x in population]
    max_fitness_value = max(all_fitness_value)
    average_fitness_value = np.mean(all_fitness_value)
    chro_max_fitness_value = population[all_fitness_value.index(max_fitness_value)]
    chro_min_fitness_value = population[all_fitness_value.index(min(all_fitness_value))]
    unique_fitness_value = len(set(all_fitness_value))
    t1 = unique_fitness_value / (size * 1.0)
    t2 = (max_fitness_value - average_fitness_value) / (max_fitness_value * 1.0)
    t3 = hamming_distance(chro_max_fitness_value, chro_min_fitness_value) / (len(chro_max_fitness_value) * 1.0)
    return t1, t2, t3


def calculate_t1_membership(t1):
    low = 0.0
    medium = 0.0
    high = 1.0

    if t1 <= 0.25:
        low = 1.0
    elif t1 <= 0.5:
        low = (-4 * t1) + 2

    if t1 <= 0.25:
        medium = 0.0
    elif t1 <= 0.5:
        medium = (4 * t1) - 1
    elif t1 <= 0.75:
        medium = (-4 * (t1 - 0.25)) + 2

    if t1 <= 0.5:
        high = 0.0
    elif t1 <= 0.75:
        high = (4 * (t1 - 0.25)) - 1

    return low, medium, high


def calculate_t2_membership(t2):
    low = 0.0
    high = 1.0

    if t2 <= 0.25:
        low = 1.0
    elif t2 <= 0.5:
        low = (-4 * t2) + 2

    if t2 <= 0.25:
        high = 0.0
    elif t2 <= 0.5:
        high = (4 * t2) - 1

    return low, high


def calculate_t3_membership(t3):
    return calculate_t1_membership(t3)


def calculate_ca_low_x(y):
    if y == 1.0:
        return 0.25
    if y == 0.0:
        return 1.0
    x = (0.25 * y) + 0.25
    return x


def calculate_ca_medium_x(y):
    if y == 0.0:
        return 0.0
    return 0.5


def calculate_ca_high_x(y):
    if y == 1.0:
        return 0.75
    if y == 0.0:
        return 0.5
    return (0.25 * y) + 0.5


def calculate_p_low_x(y):
    return calculate_ca_low_x(y)


def calculate_p_medium_x(y):
    return calculate_ca_medium_x(y)


def calculate_p_high_x(y):
    return calculate_ca_high_x(y)


def calculate_ca_and_p(population, value):
    t1, t2, t3 = calculate_t(population, value)
    t1_low, t1_medium, t1_high = calculate_t1_membership(t1)
    t2_low, t2_high = calculate_t2_membership(t2)
    t3_low, t3_medium, t3_high = calculate_t3_membership(t3)

    ca_low_array = []
    ca_medium_array = []
    ca_high_array = []

    p_low_array = []
    p_medium_array = []
    p_high_array = []

    # rule 1
    ca_high_array.append(max(t1_low, t2_low, t3_low))
    p_high_array.append(min(t1_low, t2_low, t3_low))

    # rule 2
    ca_high_array.append(max(t1_low, t2_low, t3_medium))
    p_high_array.append(min(t1_low, t2_low, t3_medium))

    # rule 3
    ca_medium_array.append(max(t1_low, t2_low, t3_high))
    p_medium_array.append(min(t1_low, t2_low, t3_high))

    # rule 4
    ca_high_array.append(max(t1_low, t2_high, t3_low))
    p_medium_array.append(min(t1_low, t2_high, t3_low))

    # rule 5
    ca_medium_array.append(max(t1_low, t2_high, t3_medium))
    p_medium_array.append(min(t1_low, t2_high, t3_medium))

    # rule 6
    ca_medium_array.append(max(t1_low, t2_high, t3_high))
    p_low_array.append(min(t1_low, t2_high, t3_high))

    # rule 7
    ca_high_array.append(max(t1_medium, t2_low, t3_low))
    p_high_array.append(min(t1_medium, t2_low, t3_low))

    # rule 8
    ca_medium_array.append(max(t1_medium, t2_low, t3_medium))
    p_medium_array.append(min(t1_medium, t2_low, t3_medium))

    # rule 9
    ca_medium_array.append(max(t1_medium, t2_low, t3_high))
    p_medium_array.append(min(t1_medium, t2_low, t3_high))

    # rule 10
    ca_medium_array.append(max(t1_medium, t2_high, t3_low))
    p_medium_array.append(min(t1_medium, t2_high, t3_low))

    # rule 11
    ca_medium_array.append(max(t1_medium, t2_high, t3_medium))
    p_medium_array.append(min(t1_medium, t2_high, t3_medium))

    # rule 12
    ca_medium_array.append(max(t1_medium, t2_high, t3_high))
    p_low_array.append(min(t1_medium, t2_high, t3_high))

    # rule 13
    ca_high_array.append(max(t1_high, t2_low, t3_low))
    p_high_array.append(min(t1_high, t2_low, t3_low))

    # rule 14
    ca_medium_array.append(max(t1_high, t2_low, t3_medium))
    p_medium_array.append(min(t1_high, t2_low, t3_medium))

    # rule 15
    ca_low_array.append(max(t1_high, t2_low, t3_high))
    p_medium_array.append(min(t1_high, t2_low, t3_high))

    # rule 16
    ca_low_array.append(max(t1_high, t2_high, t3_low))
    p_medium_array.append(min(t1_high, t2_high, t3_low))

    # rule 17
    ca_low_array.append(max(t1_high, t2_high, t3_medium))
    p_low_array.append(min(t1_high, t2_high, t3_medium))

    # rule 18
    ca_low_array.append(max(t1_high, t2_high, t3_high))
    p_low_array.append(min(t1_high, t2_high, t3_high))

    u_ca_x = sum([calculate_ca_low_x(y) * y for y in ca_low_array])
    u_ca_x += sum([calculate_ca_medium_x(y) * y for y in ca_medium_array])
    u_ca_x += sum([calculate_ca_high_x(y) * y for y in ca_high_array])
    u_ca = sum(ca_low_array)
    u_ca += sum(ca_medium_array)
    u_ca += sum(ca_high_array)
    ca = u_ca_x / (u_ca * 1.0)

    u_p_x = sum([calculate_p_low_x(y) * y for y in p_low_array])
    u_p_x += sum([calculate_p_medium_x(y) * y for y in p_medium_array])
    u_p_x += sum([calculate_p_high_x(y) * y for y in p_high_array])
    u_p = sum(p_low_array)
    u_p += sum(p_medium_array)
    u_p += sum(p_high_array)
    p = u_p_x / (u_p * 1.0)

    return ca, p

    # TODO: Crossover, mutation, elitism


def two_pc(a, b, output_array=None):
    if output_array is None:
        output_array = []
    length = len(a)
    first_point = np.random.randint(length)
    second_point = np.random.randint(length)
    if first_point > second_point:
        tmp = first_point
        first_point = second_point
        second_point = tmp

    x = []
    y = []
    for i in range(length):
        if (i < first_point) or (i > second_point):
            x.append(a[i])
            y.append(b[i])
        else:
            x.append(b[i])
            y.append(a[i])
    output_array.append(x)
    output_array.append(y)
    return output_array


def k_pc(a, b, output_array=None):
    if output_array is None:
        output_array = []

    length = len(a)
    k = np.random.randint(3, length - 1)
    k_points = random.sample(range(length), k)
    k_points = sorted(k_points)
    x = []
    y = []
    iterator = 0
    for i in range(length):
        if iterator < k:
            if i == k_points[iterator]:
                iterator += 1

        if iterator % 2 == 0:
            x.append(a[i])
            y.append(b[i])
        else:
            x.append(b[i])
            y.append(a[i])

    output_array.append(x)
    output_array.append(y)
    return output_array


def i_c(a, b, output_array=None):
    if output_array is None:
        output_array = []
    length = len(a)
    first_point = np.random.randint(length)
    second_point = np.random.randint(length)
    if first_point > second_point:
        tmp = first_point
        first_point = second_point
        second_point = tmp

    x = []
    y = []
    for i in range(length):
        if (i < first_point) or (i > second_point):
            x.append(a[i])
            y.append(b[i])
        else:
            x.append(b[length - 1 - i])
            y.append(a[length - 1 - i])
    output_array.append(x)
    output_array.append(y)
    return output_array


def crossover(population, generation, value):
    ca, p = calculate_ca_and_p(population, value)
    male, female = separate_by_gender(population, generation)
    population_size = len(population)
    total_offspring = int(round(population_size * (p * 1.0))) // 2
    offspring = []
    for x in range(total_offspring):
        a = female_tournament_selection(female, value, 5)
        b = male_selection(male, a, value)
        if ca < 0.35:
            offspring = two_pc(a, b, offspring)
        elif ca < 0.70:
            offspring = k_pc(a, b, offspring)
        else:
            offspring = i_c(a, b, offspring)
    return offspring


def mutate(offspring):
    # 1 percent chance
    chance = 3
    output_array = []
    for x in offspring:
        output_chromosome = []
        for y in x:
            random_result = np.random.randint(1, 100)
            if random_result == chance:
                output_chromosome.append((y + 1) % 2)
            else:
                output_chromosome.append(y)
        output_array.append(output_chromosome)
    return output_array
