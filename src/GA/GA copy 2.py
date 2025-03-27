import sys
import gc
sys.path.append("..")
import utils
import random
import numpy as np
from deap import base, creator, tools, algorithms
from docplex.mp.model import Model, Context
from docplex.util.status import JobSolveStatus

def evaluate(individual, task_path, net_path, piid, config_path, workers):
    print(1)
    # 从个体中解包 CPLEX 参数
    time_limit, threads, mip_gap, cut_passes = individual
    print(individual)
    # 运行 RTCSA2018，并将参数传递给该函数
    delay = RTCSA2018(task_path, net_path, piid, config_path, workers, time_limit, threads, mip_gap, cut_passes)
    return delay,

def RTCSA2018(task_path, net_path, piid, config_path, workers, time_limit, threads, mip_gap, cut_passes):
    try:

        ## ------------- LOAD DATA ------------------------------------#
        run_time = 0
        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)
        m = Model(name="RTCSA2018", log_output=False)
        # 应用传递的参数
        m.parameters.timelimit.set(time_limit)
        m.parameters.threads.set(threads)
        m.parameters.mip.tolerances.mipgap.set(mip_gap)
        m.parameters.mip.limits.cutpasses.set(cut_passes)
        m.context.cplex_parameters.threads = workers

        A = np.zeros(shape=(len(link_to_index), len(link_to_index)), dtype=int)
        for a in index_to_link:
            for b in index_to_link:
                link_a, link_b = index_to_link[a], index_to_link[b]
                if eval(link_a)[1] == eval(link_b)[0]:
                    A[a][b] = 1

        B = np.zeros(shape=(max(sw_set | es_set) + 1, len(link_to_index)),
                     dtype=int)
        for v in (sw_set | es_set):
            for e in index_to_link:
                link = eval(index_to_link[e])
                if link[0] == v:
                    B[v][e] = 1
                elif link[1] == v:
                    B[v][e] = -1

        u = m.binary_var_matrix(len(task_attr), len(link_to_index))
        t = m.integer_var_matrix(len(task_attr), len(link_to_index))

        for k in task_attr:
            for e in index_to_link:
                m.add_constraint(0 <= t[k, e])
                m.add_constraint(t[k, e] <= task_attr[k]['period'] -
                                 task_attr[k]['t_trans'])

        for f1, f2 in [(f1, f2) for f1 in task_attr for f2 in task_attr
                       if f1 < f2]:
            p1, p2 = task_attr[f1]['period'], task_attr[f2]['period']
            r1, r2 = task_attr[f1]['t_trans'], task_attr[f2]['t_trans']
            for e in index_to_link:
                _lcm = np.lcm(p1, p2)
                for a, b in [(a, b) for a in range(int(_lcm / p1))
                             for b in range(int(_lcm / p2))]:
                    m.add_constraint(
                        m.logical_or(
                            u[f1, e] == 0, u[f2, e] == 0, t[f1, e] +
                            a * p1 >= t[f2, e] + b * p2 + r2 + 1, t[f2, e] +
                            b * p2 >= t[f1, e] + a * p1 + r1 + 1) == 1)

        ## This formular in the paper is wrong
        for f in task_attr:
            m.add_constraint(
                m.sum([
                    B[task_attr[f]['src']][e] * u[f, e] for e in index_to_link
                ]) == 1)
            ## Only 1 link from ES in path
            m.add_constraint(
                m.sum(u[f, e] for e in index_to_link
                      if eval(index_to_link[e])[0] in es_set) == 1)
            ## m.add_constraint(m.sum([B[task_attr[f]['o']][e] * u[f, e] for e in index_to_link if B[task_attr[f]['o']][e] == 1]) == 1)

        ## This formular in the paper is wrong
        for f in task_attr:
            m.add_constraint(
                m.sum([
                    B[task_attr[f]['dst']][e] * u[f, e] for e in index_to_link
                ]) == -1)
            ## Only 1 link into ES in path
            m.add_constraint(
                m.sum(u[f, e] for e in index_to_link
                      if eval(index_to_link[e])[1] in es_set) == 1)
            ## m.add_constraint(m.sum([B[task_attr[f]['d']][e] * u[f, e] for e in index_to_link if B[task_attr[f]['d']][e] == -1]) == -1)

        for f in task_attr:
            for v in sw_set:
                m.add_constraint(
                    m.sum(B[v][e] * u[f, e]
                          for e in index_to_link if B[v][e] == 1) +
                    m.sum(B[v][e] * u[f, e]
                          for e in index_to_link if B[v][e] == -1) == 0)

        for ep, en in [(ep, en) for ep in index_to_link for en in index_to_link
                       if ep != en and A[ep][en] == 1]:
            for f in task_attr:
                m.add_constraint(
                    m.logical_or(
                        u[f, ep] == 0, u[f, en] == 0, t[f, en] == t[f, ep] +
                        net_attr[index_to_link[ep]]['t_proc'] +
                        task_attr[f]['t_trans'], t[f, en] +
                        task_attr[f]['period'] == t[f, ep] +
                        net_attr[index_to_link[ep]]['t_proc'] +
                        task_attr[f]['t_trans']) == 1)

        for f in task_attr:
            m.add_constraint(
                (net_attr[list(link_to_index.keys())[0]]['t_proc'] +
                 task_attr[f]['t_trans']) * m.sum(u[f, e]
                                                  for e in index_to_link) -
                net_attr[list(link_to_index.keys())[0]]['t_proc'] <=
                task_attr[f]['deadline'])

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        m.set_time_limit(utils.t_limit - utils.time_log())
        result = m.solve()
        run_time = m.solve_details.time
        run_memory = utils.mem_log()
        res = m.get_solve_status()

        if res == JobSolveStatus.UNKNOWN:
            return utils.rprint(piid, "unknown", run_time)
        elif res in [
                JobSolveStatus.INFEASIBLE_SOLUTION,
                JobSolveStatus.UNBOUNDED_SOLUTION,
                JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION
        ]:
            return utils.rprint(piid, "infeasible", run_time)

        ## GCL
        GCL = []
        for i in task_attr:
            for e_i, e in [(e, index_to_link[e]) for e in index_to_link
                           if result.get_value(u[i, e]) == 1]:
                start = int(result.get_value(t[i, e_i]))
                end = start + task_attr[i]['t_trans']
                queue = 0
                tt = task_attr[i]['period']
                for k in range(int(LCM / tt)):
                    GCL.append([
                        eval(e), queue, (start + k * tt) * utils.t_slot,
                        (end + k * tt) * utils.t_slot, LCM * utils.t_slot
                    ])
        ## Offset
        OFFSET = []
        for i in task_attr:
            start_index = np.where(B[task_attr[i]['src']] == 1)[0]
            start_index = [
                x for x in start_index if result.get_value(u[i, x]) == 1
            ][0]
            offset = int(result.get_value(t[i, start_index]))
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])

        ROUTE = []
        for i in task_attr:
            path = [
                index_to_link[e] for e in index_to_link
                if result.get_value(u[i, e]) == 1
            ]
            for link in path:
                ROUTE.append([i, eval(link)])

        QUEUE = []
        for i in task_attr:
            for e in [
                    index_to_link[e] for e in index_to_link
                    if result.get_value(u[i, e]) == 1
            ]:
                QUEUE.append([i, 0, eval(e), 0])

        DELAY = []
        for i in task_attr:
            delay = ((net_attr[list(link_to_index.keys())[0]]['t_proc'] +
                 task_attr[i]['t_trans']) * result.get_value(m.sum(u[i, e]
                                                  for e in index_to_link)) - \
                net_attr[list(link_to_index.keys())[0]]['t_proc']) * utils.t_slot
            DELAY.append([i, 0, delay])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)
        
        total_delay = 0  # 初始化总延迟变量
        for delay_entry in DELAY:
            total_delay += delay_entry[2]  # 將每個延遲的值加到總延遲中

        utils.rprint(piid, "sat", run_time)
        return total_delay  # 返回总延迟

    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)


def GA(task_path, net_path, piid, config_path="/Users/xutingwei/Desktop/tsnkit-legacy/configs/grid/0/GA", workers=1,NUM_WINDOW=5):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
# 定义遗传算法中的个体属性
    toolbox.register("attr_time_limit", random.randint, 60, 3600)  # 时间限制为 1 分钟到 1 小时
    toolbox.register("attr_threads", random.randint, 1, 12)  # 线程数
    toolbox.register("attr_mip_gap", random.uniform, 0.0, 0.01)  # MIP 间隙
    toolbox.register("attr_cut_passes", random.randint, -1, 10)  # 割平面的生成次数，-1 表示自动选择

    # 创建个体
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_time_limit, toolbox.attr_threads, toolbox.attr_mip_gap, toolbox.attr_cut_passes), n=1)

    # 合併 individual 和 population 的註冊
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 將 evaluate 函數註冊為適應度函數
    toolbox.register("evaluate", evaluate, task_path=task_path, net_path=net_path, piid=piid, config_path=config_path, workers=workers, NUM_WINDOW=NUM_WINDOW)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=1000, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # 使用 evaluate 函數進行進化算法
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 50, stats=stats, halloffame=hof, verbose=True)

    best_seed = hof[0][0]
    print("Best Z3 seed found by GA:", best_seed)
    RTCSA2018(task_path=task_path, net_path=net_path, piid=piid, config_path=config_path, workers=workers, NUM_WINDOW=NUM_WINDOW,seed=21)

if __name__ == "__main__":
    DATA = "../../data/utilization/utilization_%s_%s.csv" % (5, 1)
    TOPO = "../../data/utilization/utilization_topology.csv"
    GA(DATA, TOPO, "piid_example", "./config", 1, 5)

