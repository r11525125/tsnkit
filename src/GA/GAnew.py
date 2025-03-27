import sys
import time
import utils
import random
from deap import base, creator, tools, algorithms

# 假设的全局变量
LCM = 0
t_proc_max = 2000
GCL = {}
OFFSET = {}
ARVT = {}

# Python
# task_var 是一個字典，鍵是任務的 id，值是一個字典，包含任務的各種屬性。
task_var ={0: {'priority': 5, 'e2eD': 0, 'arr': 96, 'IT': 0, 'r': [11, 3, 2, 1, 9]}, 1: {'priority': 7, 'e2eD': 0, 'arr': 144, 'IT': 0, 'r': [9, 1, 2, 3, 4, 5, 13]}, 2: {'priority': 7, 'e2eD': 0, 'arr': 144, 'IT': 4, 'r': [13, 5, 4, 3, 2, 1, 9]}, 3: {'priority': 4, 'e2eD': 0, 'arr': 72, 'IT': 4, 'r': [9, 1, 0, 8]}, 4: {'priority': 8, 'e2eD': 0, 'arr': 168, 'IT': 0, 'r': [13, 5, 4, 3, 2, 1, 0, 8]}, 5: {'priority': 4, 'e2eD': 0, 'arr': 72, 'IT': 8, 'r': [11, 3, 2, 10]}, 6: {'priority': 4, 'e2eD': 0, 'arr': 72, 'IT': 0, 'r': [8, 0, 1, 9]}, 7: {'priority': 5, 'e2eD': 0, 'arr': 96, 'IT': 4, 'r': [11, 3, 2, 1, 9]}, 8: {'priority': 4, 'e2eD': 0, 'arr': 72, 'IT': 12, 'r': [9, 1, 2, 10]}, 9: {'priority': 5, 'e2eD': 0, 'arr': 96, 'IT': 0, 'r': [10, 2, 3, 4, 12]}} 

# task_attr 是一個字典，鍵是任務的 id，值是一個字典，包含任務的各種屬性。這個字典的結構和 task_var 類似，但是可能包含一些其他的屬性。在這個範例中，我們假設每個任務都有一個名為 "priority" 的屬性。
task_attr =  {0: {'src': 11, 'dst': 9, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [11, 3, 2, 1, 9], 's_route': ['(11, 3)', '(3, 2)', '(2, 1)', '(1, 9)'], 't_trans': 4, 'paths': [[11, 3, 2, 1, 9]]}, 1: {'src': 9, 'dst': 13, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [9, 1, 2, 3, 4, 5, 13], 's_route': ['(9, 1)', '(1, 2)', '(2, 3)', '(3, 4)', '(4, 5)', '(5, 13)'], 't_trans': 4, 'paths': [[9, 1, 2, 3, 4, 5, 13]]}, 2: {'src': 13, 'dst': 9, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [13, 5, 4, 3, 2, 1, 9], 's_route': ['(13, 5)', '(5, 4)', '(4, 3)', '(3, 2)', '(2, 1)', '(1, 9)'], 't_trans': 4, 'paths': [[13, 5, 4, 3, 2, 1, 9]]}, 3: {'src': 9, 'dst': 8, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [9, 1, 0, 8], 's_route': ['(9, 1)', '(1, 0)', '(0, 8)'], 't_trans': 4, 'paths': [[9, 1, 0, 8]]}, 4: {'src': 13, 'dst': 8, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [13, 5, 4, 3, 2, 1, 0, 8], 's_route': ['(13, 5)', '(5, 4)', '(4, 3)', '(3, 2)', '(2, 1)', '(1, 0)', '(0, 8)'], 't_trans': 4, 'paths': [[13, 5, 4, 3, 2, 1, 0, 8]]}, 5: {'src': 11, 'dst': 10, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [11, 3, 2, 10], 's_route': ['(11, 3)', '(3, 2)', '(2, 10)'], 't_trans': 4, 'paths': [[11, 3, 2, 10]]}, 6: {'src': 8, 'dst': 9, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [8, 0, 1, 9], 's_route': ['(8, 0)', '(0, 1)', '(1, 9)'], 't_trans': 4, 'paths': [[8, 0, 1, 9]]}, 7: {'src': 11, 'dst': 9, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [11, 3, 2, 1, 9], 's_route': ['(11, 3)', '(3, 2)', '(2, 1)', '(1, 9)'], 't_trans': 4, 'paths': [[11, 3, 2, 1, 9]]}, 8: {'src': 9, 'dst': 10, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [9, 1, 2, 10], 's_route': ['(9, 1)', '(1, 2)', '(2, 10)'], 't_trans': 4, 'paths': [[9, 1, 2, 10]]}, 9: {'src': 10, 'dst': 12, 'size': 1, 'period': 20000, 'deadline': 20000, 'jitter': 20000, 's_path': [10, 2, 3, 4, 12], 's_route': ['(10, 2)', '(2, 3)', '(3, 4)', '(4, 12)'], 't_trans': 4, 'paths': [[10, 2, 3, 4, 12]]}}

# net_attr 是一個字典，鍵是網絡鏈路的 id（例如 "(1, 2)"），值是一個字典，包含鏈路的各種屬性。
net_attr = {
    "(1, 2)": {"q_num": 8, "rate": 1, "t_proc": 2000, "t_prop": 0},
    "(2, 1)": {"q_num": 8, "rate": 1, "t_proc": 2000, "t_prop": 0},
}

# 假设的工具函数
def compute_delay(individual, task, path, GCL, task_attr, LCM, t_proc_max):
    # 该函数应根据个体的编码计算网络延迟
    # 返回计算得到的延迟作为适应度函数的一部分

    # 获取任务的开始时间和结束时间
    start_time = task_attr[task]['release_time']
    end_time = task_attr[task]['deadline']

    # 计算任务的执行时间
    exec_time = end_time - start_time

    # 计算任务的处理时间
    proc_time = t_proc_max

    # 计算网络延迟
    delay = exec_time - proc_time

    # 如果延迟小于0，设置为0
    if delay < 0:
        delay = 0

    return delay


def ga_optimize(task, path, GCL, task_attr, LCM, t_proc_max):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=4)  # n根据实际编码长度调整
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        delay = compute_delay(individual, task, path, GCL, task_attr, LCM, t_proc_max)
        return (delay,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    NGEN = 50
    CXPB = 0.5
    MUTPB = 0.2

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    best_ind = tools.selBest(population, 1)[0]
    return best_ind

def Scheduler(task):
    global GCL, OFFSET
    task_var[task]['arr'] = 0
    for r in task_var[task]['dst']:
        best_ind = ga_optimize(task, r, GCL, task_attr, LCM, t_proc_max)
        # 使用基因算法的结果更新GCL和OFFSET等值
        # 注意：这里假设best_ind可以直接转化为所需的设置，实际上可能需要更复杂的解码过程
        print("最优个体：", best_ind)
        # 根据best_ind进行具体的GCL、OFFSET等的更新
        # 以下代码为示例，具体实现需要根据GA的编码方式来定
        IT = int(best_ind[0] * 100)  # 假设的解码过程
        arr = int(best_ind[1] * 100)  # 假设的解码过程
        if IT == -1 or arr > task_var[task]['deadline']:
            continue
        if task_var[task]['arr'] == 0 or arr < task_var[task]['arr']:
            task_var[task]['arr'] = arr
            task_var[task]['IT'] = IT
            task_var[task]['r'] = r
        # 更新GCL、OFFSET逻辑省略，需要基于best_ind来具体实现
            
    return True

# 主函数
if __name__ == "__main__":
    tasks_to_schedule = [0, 1]  # 要调度的任务列表

    for task in tasks_to_schedule:
        Scheduler(task)

# 注意: 由于本代码依赖于众多假设和省略了很多实际应用中必要的详细实现，
# 它不能直接运行而需要你根据实际情况进行调整和完善。
