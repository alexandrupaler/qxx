from challenge_evaluation import qasm_to_dag_circuit
import networkx as nx
import math


def eval_cx_collection(cx_collection, order, limit, attenuate=False):
    sum_eval = 0

    t = 0
    skipped = 0
    for tu in cx_collection:
        c1 = order[tu[0]]
        c2 = order[tu[1]]

        if c1 > limit or c2 > limit:
            skipped += 1
            continue

        part1 = abs(c1 - c2)
        if attenuate:
            #later changes in the layout should not affect
            #the beginning of the circuit
            factor = t/len(cx_collection)
            part1 *= (1 - factor)
        sum_eval += part1

        t += 1

    if attenuate:
        # each additionally considered qubit should enable
        # as many CNOTs as possible
        # if not, penalise the cost function
        sum_eval += skipped * 200 * len(order)

    return sum_eval

'''
    Main
'''
def cuthill_order(dag_circuit):
    # qasm = ""
    # with open("./circuits/random0_n20_d20.qasm", "r") as f:
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
    parameter_qubit_increase_factor = nrq + 1#1.4

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
                sume = eval_cx_collection(cx_collection, order, limit, True)

                # print("check", qubit, "tmp sum", sume, "order", order)

                condition = sume < hold_sum
                if limit < nrq / parameter_qubit_increase_factor:
                    condition = sume > hold_sum

                if sume != prev_sum:
                    if condition:
                        hold_sum = sume
                        local_minimas.clear()
                        local_minimas.append(qubit)
                    elif sume <= hold_sum and len(local_minimas) < parameter_max_children:
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


def evaluate_leafs(all_leafs, options_tree):
    minnode = -1
    mincost = math.inf
    for nd in all_leafs:
        if options_tree.node[nd]["cost"] < mincost:
            mincost = options_tree.node[nd]["cost"]
            minnode = nd

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


