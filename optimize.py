import numpy as np
import random

from config import w_max, w_min,crossover_rate, mutation_rate, lower_bound, upper_bound, alpha, tournament_size



#### ブラックボックス最適化手法 ####
###ランダムサーチ アルゴリズム
def random_search(objective_function, bounds, n_iterations, f_RS, previous_best=None):
    # 以前の最良のスコアとパラメータを初期化
    input_history=[]
    if previous_best is None:
        best_score = float('inf')
        best_params = None
    else:
        best_params, best_score = previous_best
    for _ in range(n_iterations):
        candidate = []
        for b in bounds:
            if b[0] == 'int':
                # 整数値を生成
                value = np.random.randint(b[1], b[2] + 1)  # 上限は含まないため +1
            elif b[0] == 'float':
                # 実数値を生成
                value = np.random.uniform(b[1], b[2])
            else:
                raise ValueError(f"Unsupported type: {b[0]}")
            candidate.append(value)
        input_history.append(candidate)
        score = objective_function(candidate)
        if score < best_score:
            best_score = score
            best_params = candidate
    f_RS.write(f"\n input_history \n{input_history}")

    return best_params, best_score


###PSO アルゴリズム

# 粒子の初期化
def initialize_particles(num_particles, bounds, types):
    particles = []
    for _ in range(num_particles):
        position = [
            np.random.randint(bound[0], bound[1] + 1) if t == 'int' else np.random.uniform(bound[0], bound[1])
            for bound, t in zip(bounds, types)
        ]
        velocity = np.array([np.random.uniform(-1, 1) for _ in bounds])
        particles.append({
            'position': position,
            'velocity': velocity,
            'best_position': position.copy(),
            'best_value': float('inf'),
            'value': float('inf')
        })
    return particles


# 速度の更新
def update_velocity(particle, global_best_position, w, c1, c2):
    r1 = np.random.random(len(particle['position']))
    r2 = np.random.random(len(particle['position']))
    best_position = np.array(particle['best_position'])
    position = np.array(particle['position'])
    global_best_position = np.array(global_best_position)  # 追加: グローバルベストを NumPy 配列に変換
    cognitive = c1 * r1 * (best_position - position)
    social = c2 * r2 * (global_best_position - position)
    particle['velocity'] = w * particle['velocity'] + cognitive + social


# 位置の更新
def update_position(particle, bounds, types):
    # 位置を更新（リストなので、各要素に対して操作）
    for i in range(len(particle['position'])):
        if types[i] == 'int':
            # 実数の速度を加算後、整数にキャスト
            particle['position'][i] += particle['velocity'][i]
            particle['position'][i] = int(round(particle['position'][i]))
        else:
            # 実数の場合はそのまま加算
            particle['position'][i] += particle['velocity'][i]
        
        # 範囲外の位置を修正
        if particle['position'][i] < bounds[i][0]:
            particle['position'][i] = bounds[i][0]
        if particle['position'][i] > bounds[i][1]:
            particle['position'][i] = bounds[i][1]


# PSOアルゴリズムの実装
def PSO(objective_function, bounds, types, num_particles, num_iterations, f_PSO):
    particles = initialize_particles(num_particles, bounds, types)
    global_best_value = float('inf')
    # 初期グローバルベスト位置をランダムに設定
    global_best_position = [
        np.random.randint(bound[0], bound[1] + 1) if t == 'int' else np.random.uniform(bound[0], bound[1])
        for bound, t in zip(bounds, types)
    ]
    w = w_max      # 慣性係数
    c1 = 2.0      # 認知係数
    c2 = 2.0      # 社会係数

    result_value = np.zeros(num_iterations)
    flag_b = 0
    for iteration in range(num_iterations):
        w = w_max - (w_max - w_min) * (iteration + 1) / num_iterations
        flag_s = 0
        f_PSO.write(f'w={w}\n')
        print(f'w={w}')
        for particle in particles:
            # 目的関数の評価
            particle['value'] = objective_function(particle['position'])
            # 個々のベスト更新
            if particle['value'] < particle['best_value']:
                particle['best_value'] = particle['value']
                particle['best_position'] = particle['position'].copy()

            # グローバルベストの更新
            if particle['value'] < global_best_value:
                global_best_value = particle['value']
                global_best_position = particle['position'].copy()

            # 位置履歴の保存（最初の粒子のみ）
            if flag_s == 0:
                iteration_positions = particle['position'].copy()
                flag_s = 1
            else:
                iteration_positions = np.vstack((iteration_positions, particle['position'].copy()))
        
        if flag_b == 0:
            position_history = iteration_positions
            flag_b = 1
        else:
            position_history = np.vstack((position_history, iteration_positions))  # イテレーションごとの位置を保存

        # 各粒子の速度と位置を更新
        for particle in particles:
            update_velocity(particle, global_best_position, w, c1, c2)
            update_position(particle, bounds, types)

        result_value[iteration] = global_best_value
        print(f"Iteration {iteration + 1}/{num_iterations}, Best Value: {global_best_value}")
    
    # 位置履歴をファイルに保存
    formatted_data = '[' + ',\n '.join([str(list(row)) for row in position_history]) + ']'
    f_PSO.write(f"\nposition_history={formatted_data}")
    return global_best_position, result_value



###遺伝的アルゴリズム
def initialize_population(pop_size, lower_bound, upper_bound, types):
    """
    初期集団を生成する関数。
    
    Parameters:
        pop_size (int): 集団のサイズ。
        gene_length (int): 遺伝子の長さ（変数の数）。
        lower_bound (list): 各遺伝子の下限。
        upper_bound (list): 各遺伝子の上限。
        types (list): 各遺伝子のタイプ（'int' または 'float'）。
    
    Returns:
        np.ndarray: 初期集団。
    """
    gene_length = len(lower_bound)
    population = []
    for _ in range(pop_size):
        individual = []
        for i in range(gene_length):
            if types[i] == 'int':
                gene = np.random.randint(lower_bound[i], upper_bound[i] + 1)
            else:
                gene = np.random.uniform(lower_bound[i], upper_bound[i])
            individual.append(gene)
        population.append(individual)
    #print(population)
    return np.array(population, dtype=float)

def calculate_fitness(population, fitness_function):
    """
    集団の適応度を計算する関数。
    
    Parameters:
        population (np.ndarray): 集団。
        fitness_function (callable): 適応度関数。
    
    Returns:
        np.ndarray: 各個体の適応度。
    """
    return np.apply_along_axis(fitness_function, 1, population)

def tournament_selection(population, fitness, tournament_size):
    """
    トーナメント選択を行う関数。
    
    Parameters:
        population (np.ndarray): 集団。
        fitness (np.ndarray): 各個体の適応度。
        tournament_size (int): トーナメントのサイズ。
    
    Returns:
        np.ndarray: 選択された親集団。
    """
    selected_parents = []
    pop_size = len(population)
    for _ in range(pop_size):
        participants_idx = np.random.choice(np.arange(pop_size), tournament_size, replace=False)
        best_idx = participants_idx[np.argmin(fitness[participants_idx])]
        selected_parents.append(population[best_idx])
    return np.array(selected_parents)

def blx_alpha_crossover(parents, offspring_size, alpha, types):
    """
    BLX-α交叉を行う関数。
    
    Parameters:
        parents (np.ndarray): 親集団。
        offspring_size (tuple): 生成する子集団のサイズ。
        alpha (float): BLX-αパラメータ。
        types (list): 各遺伝子のタイプ（'int' または 'float'）。
    
    Returns:
        np.ndarray: 生成された子集団。
    """
    offspring = np.empty(offspring_size)
    gene_length = offspring_size[1]
    
    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]
        
        for gene in range(gene_length):
            if types[gene] == 'int':
                # 整数遺伝子の場合、BLX-α交叉後も整数にキャスト
                min_gene = min(parent1[gene], parent2[gene])
                max_gene = max(parent1[gene], parent2[gene])
                diff = max_gene - min_gene
                lower = min_gene - alpha * diff
                upper = max_gene + alpha * diff
                offspring[i, gene] = np.random.randint(int(lower), int(upper) + 1)
                
                if i + 1 < offspring_size[0]:
                    offspring[i + 1, gene] = np.random.randint(int(lower), int(upper) + 1)
            else:
                # 実数遺伝子の場合
                min_gene = min(parent1[gene], parent2[gene])
                max_gene = max(parent1[gene], parent2[gene])
                diff = max_gene - min_gene
                lower = min_gene - alpha * diff
                upper = max_gene + alpha * diff
                offspring[i, gene] = np.random.uniform(lower, upper)
                
                if i + 1 < offspring_size[0]:
                    offspring[i + 1, gene] = np.random.uniform(lower, upper)
    
    return offspring

def mutate(offspring, mutation_rate, lower_bound, upper_bound, types):
    """
    突然変異を行う関数。
    
    Parameters:
        offspring (np.ndarray): 子集団。
        mutation_rate (float): 突然変異率。
        lower_bound (list): 各遺伝子の下限。
        upper_bound (list): 各遺伝子の上限。
        types (list): 各遺伝子のタイプ（'int' または 'float'）。
    
    Returns:
        np.ndarray: 突然変異後の子集団。
    """
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                if types[gene_idx] == 'int':
                    offspring[idx, gene_idx] = np.random.randint(lower_bound[gene_idx], upper_bound[gene_idx] + 1)
                else:
                    offspring[idx, gene_idx] = np.random.uniform(lower_bound[gene_idx], upper_bound[gene_idx])
        # クランプ処理
        for gene_idx in range(offspring.shape[1]):
            if types[gene_idx] == 'int':
                offspring[idx, gene_idx] = int(np.clip(offspring[idx, gene_idx], lower_bound[gene_idx], upper_bound[gene_idx]))
            else:
                offspring[idx, gene_idx] = np.clip(offspring[idx, gene_idx], lower_bound[gene_idx], upper_bound[gene_idx])
    return offspring



def genetic_algorithm(objective_function, pop_size,num_generations, crossover_rate,
                      mutation_rate, lower_bound, upper_bound, alpha, tournament_size, types, f_GA):
    """
    遺伝的アルゴリズムを実行する関数。
    
    Parameters:
        objective_function (callable): 適応度関数。
        pop_size (int): 集団のサイズ。
        gene_length (int): 遺伝子の長さ（変数の数）。
        num_generations (int): 世代数。
        crossover_rate (float): 交叉率。
        mutation_rate (float): 突然変異率。
        lower_bound (list): 各遺伝子の下限。
        upper_bound (list): 各遺伝子の上限。
        alpha (float): BLX-αパラメータ。
        tournament_size (int): トーナメントサイズ。
        types (list): 各遺伝子のタイプ（'int' または 'float'）。
        f_GA (file object): 結果を記録するファイルオブジェクト。
    
    Returns:
        tuple: 最良の適応度と最良の個体。
    """
    best_fitness = float("inf")
    best_individual = None

    # 初期集団の生成
    population = initialize_population(pop_size, lower_bound, upper_bound, types)
    
    gene_history = population.copy()  # 粒子の現在位置を記録（世代数 * 集団サイズ, gene_length）

    for generation in range(num_generations):
        # 適応度の計算
        fitness = calculate_fitness(population, objective_function)

        # 現世代の最良適応度と個体の特定
        current_best_fitness = np.min(fitness)
        current_best_individual = population[np.argmin(fitness)]

        # 最良適応度の更新
        if current_best_fitness < best_fitness:
            f_GA.write(f"{current_best_fitness=},  {best_fitness=}\n")
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()

        # 現世代の情報を出力
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
        f_GA.write(f"\nGeneration {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")

        # 親の選択
        parents = tournament_selection(population, fitness, tournament_size)

        # 交叉による子集団の生成
        offspring_size = (int(pop_size * crossover_rate), len(lower_bound))
        offspring = blx_alpha_crossover(parents, offspring_size, alpha, types)

        # 突然変異の適用
        offspring = mutate(offspring, mutation_rate, lower_bound, upper_bound, types)
        
        # 子集団を集団に統合（エリート保持しない場合）
        population[0:offspring.shape[0]] = offspring

        # 位置履歴の更新
        if generation < num_generations - 1:
            gene_history = np.vstack((gene_history, population.copy()))  # 粒子の現在位置を記録

    # 位置履歴をファイルに保存
    f_GA.write(f"\ngene_history={gene_history.tolist()}")
    return best_fitness, best_individual