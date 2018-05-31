from challenge_evaluation import qasm_to_dag_circuit
import networkx as nx
import math
import operator


def sum_chain(chain, order):
    sumx = 0

    prev = order[chain[0]]
    for cd in chain[1:]:
        sumx += abs(order[cd] - prev - 1)
        prev = order[cd]

    return sumx


# def update_dictionary(dictionary, ckey):
#     if ckey not in dictionary:
#         new_name = dictionary["total"]
#         dictionary[ckey] = new_name
#         # dictionary[new_name] = ckey
#
#         dictionary["total"] += 1
#
#         # print(ckey, "is", new_name)
#
#
# def complete_dictionary(dictionary, maxnr):
#     for di in range(maxnr):
#         update_dictionary(dictionary, di)
#
#
# def get_dictionary_value(dictionary, key):
#     ret = key
#
#     if key in dictionary:
#         ret = dictionary[key]
#
#     return ret
#
#
# def get_new_qubit_names(dictionary, ckey, chain):
#     update_dictionary(dictionary, ckey)
#
#     for cd in chain:
#         update_dictionary(dictionary, cd)
#
#
# def replace_qubit_names(dictionary, chains):
#     new_chains = {}
#
#     for chaink in chains:
#         name = get_dictionary_value(dictionary, chaink)
#         new_chains[name] = []
#
#         for cv in chains[chaink]:
#             value = get_dictionary_value(dictionary, cv)
#             new_chains[name].append(value)
#
#     return new_chains
#
#
# def initialise_chains(nr_qubits):
#     ret = {}
#     for qi in range(nr_qubits):
#         ret[qi] = [qi]
#     return ret


# def analyse_chain(chain, order):
#     frequency = {}
#     for ci in chain[1:]:
#         if order[ci] not in frequency:
#             frequency[order[ci]] = 0
#         frequency[order[ci]] += 1
#
#     sumc = sum_chain(chain, order)
#
#     frequency_list = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
#
#     return {"name": order[chain[0]], "sum": sumc, "freq": frequency_list}
#
#
# def analyse_chains(chains_local, order):
#     analysis = []
#
#     for ci in chains_local:
#         analysis.append(analyse_chain(chains_local[ci], order))
#
#     analysis.sort(key=lambda x: x["sum"], reverse=True)
#
#     return analysis


# def order_swap(order, ki, kj):
#     # print("swap", ki, kj)
#
#     tmp = order[ki]
#     order[ki] = order[kj]
#     order[kj] = tmp


def eval_cx_collection(cx_collection, order, limit, attenuate=False):
    sum_eval = 0

    t = 0
    for tu in cx_collection:
        c1 = order[tu[0]]
        c2 = order[tu[1]]

        if c1 > limit or c2 > limit:
            continue

        part = abs(c1 - c2)
        if attenuate:
            factor = t/len(cx_collection)
            part *= 1-factor

        sum_eval += part

        t += 1

    return sum_eval

'''
    Main
'''
def cuthill_order(dag_circuit):
    # qasm = ""
    # with open("./circuits/random1_n20_d20.qasm", "r") as f:
    #     qasm = f.read()
    #
    # dag_circuit = qasm_to_dag_circuit(qasm)

    nrq = dag_circuit.width()
    order = list(range(nrq))

    cx_collection = []

    nodes_collection = nx.topological_sort(dag_circuit.multi_graph)
    for node in nodes_collection:
        gate = dag_circuit.multi_graph.nodes[node]
        if gate["name"] not in ["cx"]:
            continue

        q1 = int(gate["qargs"][0][1])
        q2 = int(gate["qargs"][1][1])

        cx_collection.append((q1, q2))

    sume = eval_cx_collection(cx_collection, order, nrq)
    print("naive start:", sume)

    options_tree = nx.DiGraph()
    maximum_nr_node = 0
    options_tree.add_node(maximum_nr_node, name=0, cost=0)
    maximum_nr_node += 1

    '''
        Parameters for search
    '''
    #maximum number of children of a node
    parameter_max_children = math.inf
    #maximum depth of the search tree
    parameter_max_depth = nrq
    #the first number_of_qubits * this factor the search maximises the cost
    #afterwards it minimises it
    parameter_qubit_increase_factor = 1.4

    order = [math.inf for x in order]
    for node_index in range(1, nrq):

        limit = node_index

        all_leafs = [x for x in options_tree.nodes() if options_tree.out_degree(x) == 0]

        '''
            Cut-Off search heuristic for placement
        '''
        if limit % parameter_max_depth == 0:

            minnode, mincost = evaluate_leafs(all_leafs, options_tree)

            '''
                Clean the tree and leave only the best path
            '''
            all_nodes = list(options_tree.nodes())
            all_nodes.remove(minnode)
            p_prev_leaf = minnode
            #this is the path to keep in the tree
            while len(options_tree.pred[p_prev_leaf]) == 1:
                p_prev_leaf = list(options_tree.pred[p_prev_leaf])[0]
                all_nodes.remove(p_prev_leaf)
            options_tree.remove_nodes_from(all_nodes)

            #update the leafs to be used further
            all_leafs = [minnode]


        # print("limit", limit)

        for prev_leaf in all_leafs:
            #setup ordering based on parents of this node
            set_partial_permutation(limit, options_tree, order, prev_leaf)

            #where to store candidates
            local_minimas = []
            hold_sum = math.inf
            if limit < nrq / parameter_qubit_increase_factor:
                hold_sum = - math.inf


            leaf_ancestors = [options_tree.node[x]["name"] for x in nx.ancestors(options_tree, prev_leaf)]
            leaf_ancestors.append(options_tree.node[prev_leaf]["name"])

            for qubit in range(nrq):

                if qubit in leaf_ancestors:# or qubit == options_tree.node[prev_leaf]["name"]:
                    continue

                #place qubit
                order[qubit] = limit

                prev_sum = options_tree.node[prev_leaf]["cost"]
                sume = eval_cx_collection(cx_collection, order, limit)

                # print("check", qubit, "tmp sum", sume, "order", order)

                condition = sume < hold_sum
                if limit < nrq / parameter_qubit_increase_factor:
                    condition = sume > hold_sum

                if sume != prev_sum:
                    if condition:
                        hold_sum = sume
                        local_minimas.clear()
                        local_minimas.append(qubit)
                    elif sume == hold_sum and len(local_minimas) < parameter_max_children:
                        local_minimas.append(qubit)

                # reset placement
                order[qubit] = math.inf

            #reset entire ordering
            set_partial_permutation(math.inf, options_tree, order, prev_leaf)

            # add leafs to the node that generated these permutations
            for lmin in local_minimas:
                new_node_name = maximum_nr_node
                maximum_nr_node += 1
                # print(lmin, hold_sum)
                options_tree.add_node(new_node_name, name=lmin, cost=hold_sum)
                options_tree.add_edge(prev_leaf, new_node_name)

            # print("hold", hold_qubit, "for sum", hold_sum)
            # print("local minimas", local_minimas)
            # placed_qubits.append(hold_qubit)
            # order[hold_qubit] = limit

    all_leafs = [x for x in options_tree.nodes() if options_tree.out_degree(x) == 0]
    minnode, mincost = evaluate_leafs(all_leafs, options_tree)

    set_partial_permutation(nrq, options_tree, order, minnode)

    print("sum eval:", mincost, eval_cx_collection(cx_collection, order, nrq))
    # print(order)

    return order


    # qasm = ""
    # with open("./circuits/random1_n20_d20.qasm", "r") as f:
    #     qasm = f.read()
    #
    # dag_circuit = qasm_to_dag_circuit(qasm)
    # nrq = dag_circuit.width()
    #
    # # qpic_commands = []
    # # for i in range(nrq):
    # #     wc = "q%d W" % i
    # #     qpic_commands.append(wc)
    # #     print(wc)
    #
    # chains = initialise_chains(nrq)
    #
    # nodes_collection = nx.topological_sort(dag_circuit.multi_graph)
    # for node in nodes_collection:
    #     gate = dag_circuit.multi_graph.nodes[node]
    #     if gate["name"] not in ["cx"]:
    #         continue
    #
    #     q1 = int(gate["qargs"][0][1])
    #     q2 = int(gate["qargs"][1][1])
    #
    #     chains[q1].append(q2)
    #     # chains[q2].append(q1)
    #
    #     # cx = "q%d +q%d"%(q1,q2)
    #     # qpic_commands.append(cx)
    #     # print(cx)
    #
    #
    # order = list(range(nrq))
    #
    # lastsum = math.inf
    # for tt in range(1):
    #     analysis_res = analyse_chains(chains, order)
    #
    #     evsum = sum(item['sum'] for item in analysis_res)
    #     print("evsum", evsum)
    #
    #     # if evsum <= lastsum:
    #     #     lastsum = evsum
    #     # else:
    #     #     break
    #
    #     # for x in range(len(analysis_res)):
    #     #     order[analysis_res[x]["name"]] = x
    #
    #     # xk = 0
    #     # swapped = []
    #     # for x in chains[order[analysis_res[0]["name"]]]:
    #     #     if x not in swapped:
    #     #         order_swap(order, x, xk)
    #     #         xk += 1
    #     #         swapped.append(x)
    #
    #
    #
    # print(order)
    # return order
    #
    #
    #
    # freq_matrix = []
    # for ri in range(nrq):
    #     freq_matrix.append([])
    #     for ri2 in range(nrq):
    #         freq_matrix[ri].append(0)
    #
    # for res in analysis_res:
    #     for rf in res["freq"]:
    #         freq_matrix[res["name"]][rf[0]] = rf[1]
    #
    # # for line in freq_matrix:
    # #     print(line)
    #
    # # order = list(range(nrq))
    #
    # # for tt in range(1):
    # #     for i in range(len(freq_matrix)):
    # #         j = i
    # #         while j < len(freq_matrix):
    # #             k = j + 1
    # #             while k < len(freq_matrix) - 1:
    # #                 dist1 = freq_matrix[j][k] * abs(j-k)
    # #                 dist2 = freq_matrix[j][k+1] * abs(j-k-1)
    # #
    # #                 # dist1 = freq_matrix[j][k] * abs(order[j]-order[k])
    # #                 # dist2 = freq_matrix[j][k+1] * abs(order[j]-order[k-1])
    # #
    # #                 # if freq_matrix[j][k] < freq_matrix[j][k+1]:
    # #                 if dist1 < dist2:
    # #                     # swap k/k+1
    # #
    # #                     print("swap", k, k+1)
    # #                     tmp = freq_matrix[j][k+1]
    # #                     freq_matrix[j][k + 1] = freq_matrix[j][k]
    # #                     freq_matrix[j][k] = tmp
    # #
    # #                     tmp = order[k+1]
    # #                     order[k+1] = order[k]
    # #                     order[k] = tmp
    # #                 k += 1
    # #             j += 1
    #
    # # print("------")
    # # for line in freq_matrix:
    # #     print(line)
    #
    # print(order)
    #
    # return order
    #
    # # from scipy.sparse.csgraph import reverse_cuthill_mckee
    # # from scipy.sparse import csr_matrix
    # # import numpy as np
    # #
    # # G_dense = np.array(freq_matrix)
    # # G_sparse = csr_matrix(G_dense)
    # # print("**********")
    # # print(G_sparse.A)
    # #
    # # perm = reverse_cuthill_mckee(G_sparse, True)
    # # # print("--------")
    # # # print(perm)
    # #
    # # # x = G_sparse[np.ix_(perm, perm)].A
    # # # print(x)
    # #
    # # return perm


def evaluate_leafs(all_leafs, options_tree):
    minnode = -1
    mincost = math.inf
    for nd in all_leafs:
        if options_tree.node[nd]["cost"] < mincost:
            mincost = options_tree.node[nd]["cost"]
            minnode = nd

    # print("MINNODE", minnode, mincost)

    return minnode, mincost


def set_partial_permutation(limit, options_tree, order, prev_leaf):
    position = math.inf
    if limit != math.inf:
        position = limit - 1

    order[options_tree.node[prev_leaf]["name"]] = position
    p_prev_leaf = prev_leaf
    while len(options_tree.pred[p_prev_leaf]) == 1:
        p_prev_leaf = list(options_tree.pred[p_prev_leaf])[0]
        if limit != math.inf:
            position -= 1
        order[options_tree.node[p_prev_leaf]["name"]] = position


