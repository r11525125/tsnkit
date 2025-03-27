import sys
sys.path.append("..")
import utils
import random
import numpy as np
from deap import base, creator, tools, algorithms
import gurobipy as gp
from gurobipy import GRB
import utils


def evaluate(individual, task_path, net_path, piid, config_path, workers):
    seed = individual[0]
    print(seed)
    delay = RTNS2017(task_path, net_path, piid,seed,config_path, workers)
    return delay,


def RTNS2017(task_path, net_path,piid,seed, config_path="./", workers=1):
    try:
        run_time = 0
        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)
        gp.setParam('Seed', seed)

        paths = {}
        for i in task_attr:
            paths[i] = utils.find_all_paths(net, task_attr[i]['src'],
                                            task_attr[i]['dst'])
            for k in range(len(paths[i])):
                paths[i][k] = list({
                    x: int(eval(str(paths[i][k]))[h + 1])
                    for h, x in enumerate(eval(str(paths[i][k]))[:-1])
                }.items())

        route_space = {}
        for i in paths:
            route_space[i] = set([str(x) for y in paths[i] for x in y])

        m = gp.Model(utils.myname())
        m.Params.LogToConsole = 0
        m.Params.Threads = workers
        
        x = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                      vtype=GRB.BINARY,
                      name="routing")
        t = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                      vtype=GRB.INTEGER,
                      name="time_start")

        ## Bound the t matrix
        for k in task_attr:
            for j in index_to_link:
                if index_to_link[j] in route_space[k]:
                    m.addConstr(0 <= t[k][j])
                    m.addConstr(t[k][j] <= task_attr[k]['period'] -
                                task_attr[k]['t_trans'])

        for k in task_attr:
            m.addConstr(
                gp.quicksum(x[k][link_to_index[link]]
                            for link in link_out[task_attr[k]['src']]
                            if link in route_space[k]) -
                gp.quicksum(x[k][link_to_index[link]]
                            for link in link_in[task_attr[k]['src']]
                            if link in route_space[k]) == 1)

        for k in task_attr:
            for i in (sw_set | es_set) - set(
                [task_attr[k]['src'], task_attr[k]['dst']]):
                m.addConstr(
                    gp.quicksum(x[k][link_to_index[link]]
                                for link in link_out[i]
                                if link in route_space[k]) -
                    gp.quicksum(x[k][link_to_index[link]]
                                for link in link_in[i]
                                if link in route_space[k]) == 0)

        for k in task_attr:
            for i in (sw_set | es_set):
                m.addConstr(
                    gp.quicksum(x[k][link_to_index[link]]
                                for link in link_out[i]
                                if link in route_space[k]) <= 1)
        for k in task_attr:
            for link_index in index_to_link:
                if index_to_link[link_index] in route_space[k]:
                    m.addConstr(t[k][link_index] <= utils.M * x[k][link_index])

        for k in task_attr:
            for i in (sw_set | es_set) - set(
                [task_attr[k]['src'], task_attr[k]['dst']]):
                m.addConstr(
                    gp.quicksum(t[k][link_to_index[link]]
                                for link in link_out[i]
                                if link in route_space[k]) -
                    gp.quicksum(t[k][link_to_index[link]]
                                for link in link_in[i]
                                if link in route_space[k]) >=
                    (net_attr[link_in[i][0]]['t_proc'] +
                     task_attr[k]['t_trans']) * gp.quicksum(
                         x[k][link_to_index[link]]
                         for link in link_out[i] if link in route_space[k]))

        for link in link_to_index:
            link_i = link_to_index[link]
            for k, l in [(k, l) for k in task_attr for l in task_attr
                         if k < l]:
                if link in route_space[k] and link in route_space[l]:
                    ctl, ctk = int(task_attr[l]['period']), int(
                        task_attr[k]['period'])
                    t_ijl, t_ijk = t[l][link_i], t[k][link_i]
                    rsl_k, rsl_l = task_attr[k]['t_trans'], task_attr[l][
                        't_trans']
                    x_ki, x_li = x[k][link_i], x[l][link_i]
                    lcm = int(np.lcm(ctk, ctl))
                    for u, v in [(u, v) for u in range(0, int(lcm / ctk))
                                 for v in range(0, int(lcm / ctl))]:
                        _inte = m.addVar(vtype=GRB.BINARY,
                                         name="%s%d%d%d%d" %
                                         (link, k, l, u, v))
                        m.addConstr((t_ijl + v * ctl) -
                                    (t_ijk + u * ctk) >= rsl_k - utils.M *
                                    (3 - _inte - x_ki - x_li))
                        m.addConstr((t_ijk + u * ctk) -
                                    (t_ijl + v * ctl) >= rsl_l - utils.M *
                                    (2 + _inte - x_ki - x_li))

        for k in task_attr:
            link = link_in[task_attr[k]['dst']][0]
            m.addConstr(
                gp.quicksum(t[k][link_to_index[link]]
                            for link in link_in[task_attr[k]['dst']]
                            if link in route_space[k]) -
                gp.quicksum(t[k][link_to_index[link]]
                            for link in link_out[task_attr[k]['src']]
                            if link in route_space[k]) <=
                task_attr[k]['deadline'] - task_attr[k]['t_trans'])

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)
        m.setParam('TimeLimit', utils.t_limit - utils.time_log())
        m.optimize()

        run_time = m.Runtime
        run_memory = utils.mem_log()

        if m.status == 3:
            return utils.rprint(piid, "infeasible", run_time)
        elif m.status in {9, 10, 11, 12, 16, 17}:
            return utils.rprint(piid, "unknown", run_time)

        queue_count = {}
        queue_log = {}

        ## GCL
        GCL = []
        for i in task_attr:
            period = task_attr[i]['period']
            for e_i in index_to_link:
                e = index_to_link[e_i]
                if x[i][e_i].x == 1:
                    queue_count.setdefault(e, 0)
                    start = t[i][e_i].x
                    end = start + task_attr[i]['t_trans']
                    queue = queue_count[e]
                    for k in range(int(LCM / period)):
                        GCL.append([
                            eval(e), queue,
                            int(start + k * period) * utils.t_slot,
                            int(end + k * period) * utils.t_slot,
                            LCM * utils.t_slot
                        ])
                    queue_log[(i, e)] = queue
                    queue_count[e] += 1
        ## Offset
        OFFSET = []
        for i in task_attr:
            start_link = [
                link for link in link_out[task_attr[i]['src']]
                if x[i][link_to_index[link]].x == 1
            ][0]
            offset = t[i, link_to_index[start_link]].x
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])
        ROUTE = []
        for i in task_attr:
            for k, rr in enumerate(x[i]):
                if rr.x == 1:
                    ROUTE.append([i, eval(index_to_link[k])])
        QUEUE = []
        for i in task_attr:
            for k, rr in enumerate(x[i]):
                if rr.x == 1:
                    e = index_to_link[k]
                    QUEUE.append([i, 0, eval(e), queue_log[(i, e)]])

        DELAY = []
        for i in task_attr:
            start_link = [
                link for link in link_out[task_attr[i]['src']]
                if x[i][link_to_index[link]].x == 1
            ][0]
            end_link = [
                link for link in link_in[task_attr[i]['dst']]
                if x[i][link_to_index[link]].x == 1
            ][0]
            delay = t[i, link_to_index[end_link]].x - t[
                i, link_to_index[start_link]].x + task_attr[i]['t_trans']
            DELAY.append([i, 0, delay * utils.t_slot])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)
        
        total_delay = 0  # 初始化总延迟变量
        for delay_entry in DELAY:
            total_delay += delay_entry[2]  # 將每個延遲的值加到總延遲中

        utils.rprint(piid, "sat", run_time)
        print(total_delay)
        print(x)
        print(t)
        return total_delay  # 返回总延迟
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)


def RTNS2021(task_path, net_path, piid, seed, config_path="./", workers=1):
    try:
        run_time = 0
        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        paths = {}
        for i in task_attr:
            paths[i] = utils.find_all_paths(net, task_attr[i]['src'],
                                            task_attr[i]['dst'])
            for k in range(len(paths[i])):
                paths[i][k] = list({
                    x: int(eval(str(paths[i][k]))[h + 1])
                    for h, x in enumerate(eval(str(paths[i][k]))[:-1])
                }.items())

        for i in task_attr:
            deadline = task_attr[i]['deadline']
            for path in paths[i]:
                ## We don't count the processing delay from talker
                nowait_path_time = sum([
                    task_attr[i]['t_trans'] + net_attr[str(link)]['t_proc']
                    for link in path
                ]) - net_attr[str(path[0])]['t_proc']
                if nowait_path_time > deadline:
                    paths[i].remove(path)

        route_space = {}
        for i in paths:
            route_space[i] = set([str(x) for y in paths[i] for x in y])

        m = gp.Model(utils.myname())
        m.Params.LogToConsole = 0
        m.Params.Threads = workers
        m.Params.Seed=85 

        x = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                      vtype=GRB.BINARY,
                      name="routing")
        start = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                          vtype=GRB.INTEGER,
                          name="time_start")
        end = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                        vtype=GRB.INTEGER,
                        name="time_end")

        for s in task_attr:
            m.addConstr(
                gp.quicksum(x[s][link_to_index[link]]
                            for link in link_in[task_attr[s]['src']]
                            if link in route_space[s]) == 0)

        for s in task_attr:
            m.addConstr(
                gp.quicksum(x[s][link_to_index[link]]
                            for link in link_out[task_attr[s]['src']]
                            if link in route_space[s]) == 1)
            ### Have to specify the source
            for v in es_set:
                m.addConstr(
                    gp.quicksum(x[s][link_to_index[link]]
                                for link in link_out[v]
                                if v != task_attr[s]['src']
                                and link in route_space[s]) == 0)

        for s in task_attr:
            m.addConstr(
                gp.quicksum(x[s][link_to_index[link]]
                            for link in link_out[task_attr[s]['dst']]
                            if link in route_space[s]) == 0)

        for s in task_attr:
            m.addConstr(
                gp.quicksum(x[s][link_to_index[link]]
                            for link in link_in[task_attr[s]['dst']]
                            if link in route_space[s]) == 1)

        for s in task_attr:
            for v in sw_set:
                m.addConstr(
                    gp.quicksum(x[s][link_to_index[link]]
                                for link in link_in[v]
                                if link in route_space[s]) == gp.quicksum(
                                    x[s][link_to_index[link]]
                                    for link in link_out[v]
                                    if link in route_space[s]))

        for s in task_attr:
            for v in sw_set:
                m.addConstr(
                    gp.quicksum(x[s][link_to_index[link]]
                                for link in link_out[v]
                                if link in route_space[s]) <= 1)

        for s in task_attr:
            for e in index_to_link:
                if index_to_link[e] in route_space[s]:
                    m.addConstr(end[s][e] <= task_attr[s]['period'] * x[s][e])

        for s in task_attr:
            for e in index_to_link:
                if index_to_link[e] in route_space[s]:
                    m.addConstr(end[s][e] == start[s][e] +
                                x[s][e] * task_attr[s]['t_trans'])

        for s in task_attr:
            for v in sw_set:
                m.addConstr(
                    gp.quicksum(end[s][link_to_index[e]] +
                                x[s][link_to_index[e]] * net_attr[e]['t_proc']
                                for e in link_in[v]
                                if e in route_space[s]) == gp.quicksum(
                                    start[s][link_to_index[e]]
                                    for e in link_out[v]
                                    if e in route_space[s]))

        for s, s_p in [(s, s_p) for s in task_attr for s_p in task_attr
                       if s < s_p]:
            s_t, s_p_t = task_attr[s]['period'], task_attr[s_p]['period']
            lcm = np.lcm(s_t, s_p_t)
            for e in index_to_link:
                if index_to_link[e] in route_space[s] and index_to_link[
                        e] in route_space[s_p]:
                    for a, b in [(a, b) for a in range(0, int(lcm / s_t))
                                 for b in range(0, int(lcm / s_p_t))]:
                        _inte = m.addVar(vtype=GRB.BINARY,
                                         name="%d%d%s" %
                                         (s, s_p, index_to_link[e]))
                        m.addConstr(
                            end[s][e] + a * s_t <= start[s_p][e] - 1 +
                            b * s_p_t +
                            (2 + _inte - x[s][e] - x[s_p][e]) * utils.M)
                        m.addConstr(
                            end[s_p][e] + b * s_p_t <= start[s][e] - 1 +
                            a * s_t +
                            (3 - _inte - x[s][e] - x[s_p][e]) * utils.M)

        for s in task_attr:
            start_t = gp.quicksum(start[s][link_to_index[e]]
                                  for e in link_out[task_attr[s]['src']]
                                  if e in route_space[s])
            end_t = gp.quicksum(end[s][link_to_index[dst_e]] for dst_e in [
                link for link in link_in[task_attr[s]['dst']]
                if link in route_space[s]
            ])
            m.addConstr(end_t - start_t <= task_attr[s]['deadline'])

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        m.setParam('TimeLimit', utils.t_limit - utils.time_log())
        m.optimize()
        run_time = m.Runtime
        run_memory = utils.mem_log()

        if m.status == 3:
            return utils.rprint(piid, "infeasible", run_time)
        elif m.status in {9, 10, 11, 12, 16, 17}:
            return utils.rprint(piid, "unknown", run_time)

        ## GCL
        GCL = []
        for i in task_attr:
            period = task_attr[i]['period']
            for e_i in index_to_link:
                link = index_to_link[e_i]
                if x[i][e_i].x > 0:
                    s = start[i][e_i].x
                    e = end[i][e_i].x
                    queue = 0
                    for k in range(int(LCM / period)):
                        GCL.append([
                            eval(link), 0,
                            int(s + k * period) * utils.t_slot,
                            int(e + k * period) * utils.t_slot,
                            LCM * utils.t_slot
                        ])
        ## Offset
        OFFSET = []
        for i in task_attr:
            start_link = [
                link for link in link_out[task_attr[i]['src']]
                if x[i][link_to_index[link]].x > 0
            ][0]
            offset = start[i, link_to_index[start_link]].x
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])

        ROUTE = []
        for i in task_attr:
            for k, rr in enumerate(x[i]):
                if rr.x > 0:
                    ROUTE.append([i, eval(index_to_link[k])])

        QUEUE = []
        for i in task_attr:
            for k, rr in enumerate(x[i]):
                if rr.x > 0:
                    e = index_to_link[k]
                    QUEUE.append([i, 0, eval(e), 0])
        DELAY = []
        for i in task_attr:
            start_link = [
                link for link in link_out[task_attr[i]['src']]
                if x[i][link_to_index[link]].x > 0
            ][0]
            end_link = [
                link for link in link_in[task_attr[i]['dst']]
                if x[i][link_to_index[link]].x > 0
            ][0]
            DELAY.append([
                i, 0,
                (end[i, link_to_index[end_link]].x -
                 start[i, link_to_index[start_link]].x) * utils.t_slot
            ])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)


        total_delay = 0  # 初始化总延迟变量
        for delay_entry in DELAY:
            total_delay += delay_entry[2]  # 將每個延遲的值加到總延遲中

        utils.rprint(piid, "sat", run_time)
        print(total_delay)
        return total_delay  # 返回总延迟
    
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)



def GA(task_path, net_path, piid, config_path="/Users/xutingwei/Desktop/tsnkit-legacy/configs/grid/0/GA", workers=1,NUM_WINDOW=5):
    print(1)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 1000) # 随机种子的范围
    
    # 合併 individual 和 population 的註冊
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 將 evaluate 函數註冊為適應度函數
    toolbox.register("evaluate", evaluate, task_path=task_path, net_path=net_path, piid=piid, config_path=config_path, workers=workers)
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
    RTNS2017(task_path=task_path, net_path=net_path, piid=piid, config_path=config_path, workers=workers, seed=best_seed)

if __name__ == "__main__":
    DATA = "../../data/utilization/utilization_%s_%s.csv" % (5, 1)
    TOPO = "../../data/utilization/utilization_topology.csv"
    GA(DATA, TOPO, "piid_example", "./config", 1, 5)

