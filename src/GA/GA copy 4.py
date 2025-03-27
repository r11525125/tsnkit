import sys
import traceback

sys.path.append("..")
import time
import utils
import random
from deap import base, creator, tools, algorithms

random.seed(0)



def match_time(t, sche) -> int:
    '''
    Use binary search to quickly find the posistion of GCL
    '''
    if not sche:
        return -1
    gate_time = [x[0] for x in sche]
    left = 0
    right = len(sche) - 1
    if gate_time[right] <= t < sche[-1][1]:
        return right
    elif sche[-1][1] <= t:
        return -2
    elif t < gate_time[0]:
        return -1

    while True:
        median = (left + right) // 2
        if right - left <= 1:
            return left
        elif gate_time[left] <= t < gate_time[median]:
            right = median
        else:
            left = median


def FindET(task, GCL, path, start_time_option=None):
    arr = (len(path) - 1) * (t_proc_max + task_attr[task]['t_trans'])
    task_p = task_attr[task]['period']
    feasible_starts = []  # 存储所有可行的开始时间

    for it in range(0, task_p - arr + 1):
        _last_hop_end = it
        for i, v in enumerate(path[:-1]):
            link_flag = True
            link = str((v, path[i + 1]))
            _current_hop_start = _last_hop_end
            _last_hop_end = _current_hop_start + t_proc_max + task_attr[task]['t_trans']
            for alpha in range(0, int(LCM / task_p)):
                match_start = match_time(_current_hop_start + alpha * task_p, GCL[link])
                match_end = match_time(_last_hop_end - t_proc_max + alpha * task_p, GCL[link])
                if match_start == -1 and GCL[link] and _last_hop_end - t_proc_max + alpha * task_p > GCL[link][0][0]:
                    link_flag = False
                    break
                if match_start == -2 and GCL[link] and _current_hop_start + alpha * task_p < GCL[link][-1][1]:
                    link_flag = False
                    break
                if match_start >= 0 and (match_start != match_end or (GCL[link] and GCL[link][match_start][1] > _current_hop_start + alpha * task_p)):
                    link_flag = False
                    break

            if not link_flag:
                break  # 如果这个链接不可用，检查下一个时间点
        else:
            feasible_starts.append(it)  # 所有链接都可行，添加到可行开始时间列表

    if feasible_starts:
        if start_time_option is not None and start_time_option < len(feasible_starts):
            # 选择最接近start_time_option的开始时间
            selected_start = min(feasible_starts, key=lambda x: abs(x - start_time_option))
            #print(selected_start,start_time_option)
        else:
            selected_start = random.choice(feasible_starts)  # 如果没有提供start_time_option，则随机选择
        #print(f"Task {task},feasible_starts:{start_time_option},selected start time: {selected_start*utils.t_slot}")
        return selected_start
    else:
        return -1


def Scheduler(task, start_time_choice, selected_path_index):
    global GCL, OFFSET, ARVT
    # 选定的路径
    selected_path = task_attr[task]['paths'][selected_path_index]
    
    task_var[task]['arr'] = 0
    # 使用选定的路径和起始时间选项进行调度
    IT = FindET(task, GCL, selected_path, start_time_choice)
    if IT == -1:
        return False  # 如果没有找到可行的开始时间，返回False

    arr = (len(selected_path) - 1) * (t_proc_max + task_attr[task]['t_trans'])
    task_var[task]['arr'] = arr
    task_var[task]['IT'] = IT
    task_var[task]['r'] = selected_path

    _last_hop_end = IT
    for i, v in enumerate(selected_path[:-1]):
        link = str((v, selected_path[i + 1]))
        _current_hop_start = _last_hop_end
        _last_hop_end = _current_hop_start + t_proc_max + task_attr[task]['t_trans']
        for alpha in range(0, int(LCM / task_attr[task]['period'])):
            GCL[link].append([_current_hop_start + alpha * task_attr[task]['period'], _last_hop_end - t_proc_max + alpha * task_attr[task]['period'], 0])

        if i == 0:
            OFFSET[task] = _current_hop_start
        if i == len(selected_path) - 2:
            ARVT[task] = _last_hop_end - t_proc_max

    return True



def GA(task_path, net_path, piid, config_path="./", workers=1):
    try:
        global GCL, OFFSET, ARVT, task_var, task_attr, net_attr, LCM, t_proc_max
        net, net_attr, _, _, link_to_index, _, _, _, rate = utils.read_network(net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)
        t_proc_max = max(net_attr[link]['t_proc'] for link in net_attr)

        task_var = {k: {'priority': 0, 'e2eD': 0, 'arr': 0, 'IT': 0, 'r': None} for k in task_attr}
        for k in task_attr:
            task_attr[k]['paths'] = utils.find_all_paths(net, task_attr[k]['src'], task_attr[k]['dst'])
            task_var[k]['priority'] = max(len(x) for x in task_attr[k]['paths'])

        task_order = sorted(task_attr.keys(), key=lambda x: task_var[x]['priority'], reverse=True)

        # 更新：为每个任务的路径选择添加逻辑
        num_paths_per_task = [len(task_attr[task]['paths']) for task in task_order]
        total_num_choices = sum(num_paths_per_task)  # 总路径选择数

        pop_size = 100  # 更新：种群大小
        num_generations = 50  # 更新：代数

        toolbox = base.Toolbox()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # 更新：为每个任务定义属性
        def create_individual():
            start_times = [random.randint(0, task_attr[task_order[i]]['period']) for i in range(len(task_order))]
            path_indices = [random.randint(0, num_paths - 1) for num_paths in num_paths_per_task]
            return creator.Individual(start_times + path_indices)

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 更新：评估函数
        def evaluate(individual):
            GCL.clear()
            for link in link_to_index: GCL.setdefault(link, [])
            OFFSET.clear()
            ARVT.clear()
            start_time_choices = individual[:len(task_order)]
            path_indices = individual[len(task_order):]
            for i, task in enumerate(task_order):
                if not Scheduler(task, start_time_choices[i], path_indices[i]):
                    return float('inf'),  # 无法调度
            makespan = max(ARVT.values()) - min(OFFSET.values())
            return 1 / makespan,

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=max(task_attr[task_order[0]]['period'], max(num_paths_per_task)-1), indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)

        for gen in range(num_generations):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring): ind.fitness.values = fit
            pop = toolbox.select(offspring, k=len(pop))

            best_ind = tools.selBest(pop, k=1)[0]
            print(f"Generation {gen}: Best start time choices: {best_ind}, Best makespan: {1/best_ind.fitness.values[0]}")

        best_ind = tools.selBest(pop, k=1)[0]
        print(f"Best overall start time choices: {best_ind}, Best overall makespan: {1/best_ind.fitness.values[0]}")

        # 运行最佳个体的调度以生成最终的调度结果
        start_time_choices = best_ind[:len(task_order)]
        path_indices = best_ind[len(task_order):]
        GCL.clear()
        for link in link_to_index: GCL.setdefault(link, [])
        OFFSET.clear()
        ARVT.clear()
        for i, task in enumerate(task_order):
            Scheduler(task, start_time_choices[i], path_indices[i])
            if not success:
                return utils.rprint(piid, 'infeasible', 0)  # Return infeasible if unable to schedule
            if ARVT[task] - OFFSET[task] > utils.t_limit:
                #print(ARVT[task]-OFFSET[task])
                return utils.rprint(piid, 'unknown', 0)  # Return unknown if exceeds time limit
        # Output results
        GCL_out = []
        for link in GCL:
            [
                GCL_out.append([
                    eval(link), row[2], row[0] * utils.t_slot,
                    row[1] * utils.t_slot, LCM * utils.t_slot
                ]) for row in GCL[link]
            ]
        GCL = GCL_out

        OFFSET_out = []
        #print(OFFSET)
        for i in OFFSET:
            OFFSET_out.append(
                [i, 0, (task_attr[i]['period'] - OFFSET[i]) * utils.t_slot])
        #print(OFFSET_out)

        ROUTE = []
        for i in task_var:
            route = task_var[i]['r']
            for h, v in enumerate(route[:-1]):
                ROUTE.append([i, (v, route[h + 1])])

        QUEUE = []
        for i in task_var:
            route = task_var[i]['r']
            for h, v in enumerate(route[:-1]):
                QUEUE.append([i, 0, (v, route[h + 1]), 0])

        DELAY = []
        for i in task_attr:
            DELAY.append([i, 0, (ARVT[i] - OFFSET[i]) * utils.t_slot])
        
        #print(ARVT,OFFSET)
        utils.write_result(utils.myname(), piid, GCL, OFFSET_out, ROUTE, QUEUE,
                           DELAY, config_path)
        
        return utils.rprint(piid, "sat", 0)  # Return success
        
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", 0)  # Return unknown if interrupted
    except Exception as e:
        print(e)
