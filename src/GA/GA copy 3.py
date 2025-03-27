import sys
sys.path.append("..")
import z3
import utils
import random
import numpy as np
from deap import base, creator, tools, algorithms


def evaluate(individual, task_path, net_path, piid, config_path, workers, U):
    # 使用个体中的种子作为随机种子运行 RTAS2018
    print(individual)
    seed = individual[0]
    # 运行 RTAS2018，返回总延迟作为适应度评价

    delay = RTAS2018(task_path, net_path, piid, config_path, workers, U, seed)
    return delay,

def ASPDAC2022(task_path, net_path, piid, config_path="./", workers=1, U=3,seed=0):
    try:
        run_time = 0
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)
        z3.set_param('parallel.enable', True)
        z3.set_param('parallel.threads.max', workers)
        z3.set_param('smt.random_seed', seed)

        s = z3.Solver()

        task_var = {}
        ## Assume task is strictly periodic
        for i in task_attr:
            task_var.setdefault(i, {})
            t_period = task_attr[i]['period']
            route = task_attr[i]['s_path']
            for _i, a in enumerate(route[:-1]):
                link = str((a, route[_i + 1]))
                task_var[i].setdefault(link, {})

                task_var[i][link]['N'] = z3.BoolVector(
                    "n_" + str(i) + '_' + str(link), int(LCM / t_period))

                for j in range(0, int(LCM / t_period)):
                    task_var[i][link].setdefault(j, {})
                    task_var[i][link][j]['r'] = z3.IntVector(
                        "r_" + str(i) + str(j) + '_' + str(link), U)
                    task_var[i][link][j]['f'] = z3.IntVector(
                        "f_" + str(i) + str(j) + '_' + str(link), U)
                    s.add(
                        j * t_period <= task_var[i][link][j]['r'][0],
                        task_var[i][link][j]['f'][U - 1] <= (j + 1) * t_period)

                    s.add(
                        z3.Or(task_var[i][link]['N'][j] == True,
                              task_var[i][link]['N'][j] == False))

        for i, k in [(i, k) for i in task_var for k in task_var if i != k]:
            for link in [
                    link for link in net_attr
                    if link in task_var[i] and link in task_var[k]
            ]:
                for j, l in [(j, l)
                             for j in range(int(LCM / task_attr[i]['period']))
                             for l in range(int(LCM / task_attr[k]['period']))
                             ]:
                    ## Situation AD in the paper, no preemption
                    s.add(
                        ## There is no constraints on preemption level
                        z3.Implies(
                            task_var[i][link]['N'][j] == task_var[k][link]['N']
                            [l],
                            z3.Or(
                                task_var[i][link][j]['f'][U - 1] <=
                                task_var[k][link][l]['r'][0],
                                task_var[k][link][l]['f'][U - 1] <=
                                task_var[i][link][j]['r'][0])))

                    ## Situation BC in the paper, i(express), k(preemptive)
                    s.add(
                        z3.Implies(
                            ## Two levels
                            z3.And(task_var[i][link]['N'][j] == True,
                                   task_var[k][link]['N'][l] == False),
                            ## Preemption can happen between 1 and U-1
                            z3.Or([
                                z3.And(
                                    ## Start equal to the end of the express frame
                                    task_var[k][link][l]['r'][y] == task_var[i]
                                    [link][j]['f'][U - 1],
                                    ## End of last segment equal to the start of the preemptive frame
                                    task_var[k][link][l]['f'][
                                        y - 1] == task_var[i][link][j]['r'][0],
                                ) for y in range(1, U)
                            ] + [
                                z3.Or(
                                    task_var[i][link][j]['f'][
                                        U - 1] <= task_var[k][link][l]['r'][0],
                                    task_var[k][link][l]['f'][
                                        U - 1] <= task_var[i][link][j]['r'][0])
                            ])))

                    # temp = []
                    # ## No preemption for Ethernet header
                    # temp.append(
                    #     z3.Or(
                    #         task_var[i][link][j]['f'][0] <=
                    #         task_var[k][link][l]['r'][0],
                    #         task_var[k][link][l]['f'][0] <=
                    #         task_var[i][link][j]['r'][0]))
                    # ## Either the fragment is not active or the fragment is adjacent to the express frame
                    # for y in [y for y in range(1, U)]:
                    #     temp.append(
                    #         z3.Or(
                    #             task_var[k][link][l]['f'][y] -
                    #             task_var[k][link][l]['r'][y] == 0,
                    #             z3.And(
                    #                 task_var[k][link][l]['r'][y] == task_var[i]
                    #                 [link][j]['f'][-1],
                    #                 task_var[k][link][l]['f'][
                    #                     y - 1] == task_var[i][link][j]['r'][0],
                    #             )))
                    # ## Applied when two tasks are in different preemption levels
                    # s.add(
                    #     z3.Implies(
                    #         z3.And(task_var[i][link]['N'][j] == True,
                    #                task_var[k][link]['N'][l] == False),
                    #         z3.And(temp)))
                    ## test #############################################################

        # res = s.check()
        # print("[2] ", res)
        for i in task_var:
            for link in task_var[i]:
                for j in range(int(LCM / task_attr[i]['period'])):
                    s.add(
                        z3.Implies(
                            task_var[i][link]['N'][j] == True,
                            z3.And(task_var[i][link][j]['f'][U - 1] -
                                   task_var[i][link][j]['r'][0] == task_attr[i]
                                   ['t_trans'])),
                        z3.Implies(
                            task_var[i][link]['N'][j] == False,
                            z3.Sum([(task_var[i][link][j]['f'][z] -
                                     task_var[i][link][j]['r'][z])
                                    for z in range(0, U)
                                    ]) == task_attr[i]['t_trans'],
                        ),
                        # z3.Implies(
                        # task_var[i][link]['N'][j] == False,
                        z3.And([
                            z3.And(
                                task_var[i][link][j]['r'][p] <=
                                task_var[i][link][j]['f'][p],
                                task_var[i][link][j]['f'][p] <=
                                task_var[i][link][j]['r'][p + 1],
                                task_var[i][link][j]['r'][p + 1] <=
                                task_var[i][link][j]['f'][p + 1])
                            for p in range(0, U - 1)
                        ]),
                    )
        ## test ##
        # res = s.check()
        # print("[3] ", res)
        for i in task_var.keys():
            path = list(task_var[i].keys())
            for _i, link in enumerate(path[:-1]):
                next_hop = path[_i + 1]
                for j in range(int(LCM / task_attr[i]['period'])):
                    s.add(task_var[i][next_hop][j]['r'][0] >=
                          task_var[i][link][j]['f'][U - 1] +
                          net_attr[link]['t_proc'])
                ## test ##
        # res = s.check()
        # print("[4] ", res)
        for i in task_var.keys():
            _hop_s = list(task_var[i].items())[0]
            _hop_e = list(task_var[i].items())[-1]
            for a in range(int(LCM / task_attr[i]['period'])):
                s.add(_hop_s[1][a]['r'][0] +
                      task_attr[i]['deadline'] >= _hop_e[1][a]['f'][U - 1] +
                      utils.delta)

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        s.set("timeout", int(utils.t_limit - utils.time_log()) * 1000)
        res = s.check()
        # print("[5] ", res)

        info = s.statistics()
        run_time = info.time
        # run_memory = info.max_memory
        # run_memory = utils.mem_log()

        # print(res)

        if res == z3.unsat:
            return utils.rprint(piid, "infeasible", run_time)
        elif res == z3.unknown:
            return utils.rprint(piid, "unknown", run_time)

        result = s.model()

        ## GCL
        queue_count = {}
        queue_log = {}
        GCL = []
        for i in task_var:
            for e in task_var[i]:
                queue_count.setdefault(e, 0)
                for j in range(int(LCM / task_attr[i]['period'])):
                    # print(str(result[task_var[i][e]['N'][j]]))
                    start = result[task_var[i][e][j]['r'][0]].as_long()
                    end = result[task_var[i][e][j]['f'][-1]].as_long()
                    queue = queue_count[e]
                    GCL.append([
                        eval(e), queue, (start) * utils.t_slot,
                        (end) * utils.t_slot, LCM * utils.t_slot
                    ])
                queue_log[(i, e)] = queue
                queue_count[e] += 1

        OFFSET = []
        for i in task_var:
            e = list(task_var[i].keys())[0]
            for j in range(int(LCM / task_attr[i]['period'])):
                OFFSET.append([
                    i, 0,
                    (task_attr[i]['period'] -
                     result[task_var[i][e][j]['r'][0]].as_long()) *
                    utils.t_slot
                ])

        ROUTE = []
        for i in task_var:
            route = list(task_var[i].keys())
            for x in route:
                ROUTE.append([i, eval(x)])

        QUEUE = []
        for i in task_var:
            for e in list(task_var[i].keys()):
                QUEUE.append([i, 0, e, queue_log[(i, e)]])

        ## Log the delay

        DELAY = []
        for i in task_var:
            _hop_s = list(task_var[i].items())[0]
            _hop_e = list(task_var[i].items())[-1]
            for j in range(int(LCM / task_attr[i]['period'])):
                DELAY.append([
                    i, j,
                    (result[_hop_e[1][j]['f'][-1]].as_long() -
                     result[_hop_s[1][j]['r'][0]].as_long() +
                     net_attr[_hop_e[0]]['t_proc']) * utils.t_slot
                ])
        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        return utils.rprint(piid, "sat", run_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)

def RTAS2018(task_path, net_path, piid, config_path, workers, NUM_WINDOW, seed):
    try:
        run_time = 0
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)
        print(seed)
        s = z3.Solver()
        z3.set_param('smt.random_seed', 21)
        z3.set_param('parallel.enable', False)
        z3.set_param('parallel.threads.max', workers)

        net_var = {}
        for link in net_attr:
            net_var.setdefault(link, {})
            net_attr[link]['W'] = NUM_WINDOW
            net_var[link]['phi'] = z3.Array(link + '_' + 'phi', z3.IntSort(),
                                            z3.IntSort())
            net_var[link]['tau'] = z3.Array(link + '_' + 'tau', z3.IntSort(),
                                            z3.IntSort())
            net_var[link]['k'] = z3.Array(link + '_' + 'k', z3.IntSort(),
                                          z3.IntSort())

        task_var = {}

        for i in task_attr:
            task_var.setdefault(i, {})
            route = task_attr[i]['s_path']
            for _i, a in enumerate(route[:-1]):
                link = str((a, route[_i + 1]))
                task_var[i].setdefault(link, {})
                task_var[i][link] = []
                for j in range(int(LCM / task_attr[i]['period'])):
                    task_var[i][link].append(
                        z3.Int('w_' + str(i) + '_' + str(link) + '_' + str(j)))

        for link in net_var:
            for k in range(net_attr[link]['W']):
                net_var[link]['tau'] = z3.Store(net_var[link]['tau'], k,
                                                net_var[link]['phi'][k])

        for i in task_var:
            for link in task_var[i]:
                for j in task_var[i][link]:
                    net_var[link]['tau'] = z3.Store(
                        net_var[link]['tau'], j,
                        net_var[link]['tau'][j] + task_attr[i]['t_trans'])

        for link in net_var:
            s.add(net_var[link]['phi'][0] >= 0, net_var[link]['tau'][-1] < LCM)

        for link in net_var:
            for k in range(net_attr[link]['W']):
                s.add(net_var[link]['k'][k] >= 0,
                      net_var[link]['k'][k] < net_attr[link]['q_num'])

        for i in task_var:
            for link in task_var[i]:
                for j in range(int(LCM / task_attr[i]['period'])):
                    s.add(
                        net_var[link]['phi'][task_var[i][link][j]] >=
                        j * task_attr[i]['period'],
                        net_var[link]['tau'][task_var[i][link][j]] <
                        (j + 1) * task_attr[i]['period'])

        for link in net_var:
            for i in range(net_attr[link]['W'] - 1):
                s.add(net_var[link]['tau'][i] <= net_var[link]['phi'][i + 1])

        for i in task_var:
            for link in task_var[i]:
                for j in task_var[i][link]:
                    s.add(0 <= j, j < net_attr[link]['W'])

        for i in task_var:
            hops = list(task_var[i].keys())
            for k, link in enumerate(hops[:-1]):
                for j in range(int(LCM / task_attr[i]['period'])):
                    s.add(net_var[link]['tau'][task_var[i][link][j]] +
                          net_attr[link]['t_proc'] + utils.delta <= net_var[
                              hops[k + 1]]['phi'][task_var[i][hops[k + 1]][j]])

        for i, j in [(i, j) for i in task_var for j in task_var if i < j]:
            path_i = list(task_var[i].keys())
            path_j = list(task_var[j].keys())
            for x_a, y_a, a_b in [(path_i[_x - 1], path_j[_y - 1], i_a_b)
                                  for _x, i_a_b in enumerate(path_i)
                                  for _y, j_a_b in enumerate(path_j)
                                  if i_a_b == j_a_b]:
                for k, l in [(k, l)
                             for k in range(int(LCM / task_attr[i]['period']))
                             for l in range(int(LCM / task_attr[j]['period']))
                             ]:
                    s.add(
                        z3.Or(
                            net_var[a_b]['tau'][task_var[i][a_b][k]] +
                            net_attr[y_a]['t_proc'] + utils.delta <
                            net_var[y_a]['phi'][task_var[j][y_a][l]],
                            net_var[a_b]['tau'][task_var[j][a_b][l]] +
                            net_attr[x_a]['t_proc'] + utils.delta <
                            net_var[x_a]['phi'][task_var[i][x_a][k]],
                            net_var[a_b]['k'][task_var[i][a_b][k]] !=
                            net_var[a_b]['k'][task_var[j][a_b][l]],
                            task_var[i][a_b][k] == task_var[j][a_b][l]))

        for i in task_var:
            _hop_s = list(task_var[i].keys())[0]
            _hop_e = list(task_var[i].keys())[-1]
            for j in range(int(LCM / task_attr[i]['period'])):
                s.add(net_var[_hop_e]['tau'][task_var[i][_hop_e][j]] -
                      net_var[_hop_s]['phi'][task_var[i][_hop_s][j]] <=
                      task_attr[i]['deadline'] - utils.delta)

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        s.set("timeout", int(utils.t_limit - utils.time_log()) * 1000)
        res = s.check()
        info = s.statistics()
        run_time = info.time
        # run_memory = info.max_memory

        if res == z3.unsat:
            return utils.rprint(piid, "infeasible", run_time)
        elif res == z3.unknown:
            return utils.rprint(piid, "unknown", run_time)
        result = s.model()

        ## GCL
        GCL = []
        for link in net_var:
            for i in range(net_attr[link]['W']):
                start = result.eval(net_var[link]['phi'][i]).as_long()
                end = result.eval(net_var[link]['tau'][i]).as_long()
                queue = result.eval(net_var[link]['k'][i]).as_long()
                if end > start:
                    GCL.append([
                        eval(link), queue, start * utils.t_slot,
                        end * utils.t_slot, LCM * utils.t_slot
                    ])

        ## Offset
        OFFSET = []
        for i in task_var:
            link = list(task_var[i].keys())[0]
            for ins_id, ins_window in enumerate(task_var[i][link]):
                offset = result.eval(
                    net_var[link]['phi'][ins_window]).as_long()
                OFFSET.append([
                    i, ins_id, (task_attr[i]['period'] - offset) * utils.t_slot
                ])

        ROUTE = []
        for i in task_attr:
            route = task_attr[i]['s_path']
            for h, v in enumerate(route[:-1]):
                ROUTE.append([i, (v, route[h + 1])])

        QUEUE = []
        for i in task_var:
            for link in task_var[i]:
                for ins_id, ins_window in enumerate(task_var[i][link]):
                    QUEUE.append([
                        i, ins_id, link,
                        result.eval(net_var[link]['k'][ins_window]).as_long()
                    ])

        DELAY = []
        for i in task_var:
            link_start = list(task_var[i].keys())[0]
            link_end = list(task_var[i].keys())[-1]
            for ins_id in range(int(LCM / task_attr[i]['period'])):
                start_window = task_var[i][link_start][ins_id]
                start = result.eval(
                    net_var[link_start]['phi'][start_window]).as_long()
                end_window = task_var[i][link_end][ins_id]
                end = result.eval(
                    net_var[link_end]['tau'][end_window]).as_long()
                delay = (end - start) * utils.t_slot
                DELAY.append([i, ins_id, delay])


        total_delay = 0  # 初始化总延迟变量
        for delay_entry in DELAY:
            total_delay += delay_entry[2]  # 將每個延遲的值加到總延遲中
        
        print(f"Total delay: {total_delay} time units")   

        utils.write_result("GA", piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)
            # 遍历所有任务计算总延迟
        
        
        return total_delay  # 返回总延迟
    except Exception as e:
        print(f"An error occurred: {e}")



def GA(task_path, net_path, piid, config_path="/Users/xutingwei/Desktop/tsnkit-legacy/configs/grid/0/GA", workers=1,U=3):
    print(1)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 1) # 随机种子的范围
    
    # 合併 individual 和 population 的註冊
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 將 evaluate 函數註冊為適應度函數
    toolbox.register("evaluate", evaluate, task_path=task_path, net_path=net_path, piid=piid, config_path=config_path, workers=workers, U=U)
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
    RTAS2018(task_path=task_path, net_path=net_path, piid=piid, config_path=config_path, workers=workers, NUM_WINDOW=NUM_WINDOW,seed=21)

if __name__ == "__main__":
    DATA = "../../data/utilization/utilization_%s_%s.csv" % (5, 1)
    TOPO = "../../data/utilization/utilization_topology.csv"
    GA(DATA, TOPO, "piid_example", "./config", 1, 5)

