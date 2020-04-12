import random

import networkx as nx
import math


def get_distance_coupling(c1, c2, coupling_object):
    """

    :param c1:
    :param c2:
    :param coupling_object:
    :return:
    """

    idx1 = coupling_object.coupling.physical_qubits[c1]
    idx2 = coupling_object.coupling.physical_qubits[c2]

    dist = coupling_object.coupling_dist[idx1][idx2]

    return dist


def get_distance_linear(c1, c2):
    """

    :param c1:
    :param c2:
    :return:
    """
    return abs(c1 - c2)


def get_distance_offsets(circ_q1, circ_q2, offsets, local_circuit_to_phys, coupling_object):

    minq = min(circ_q1, circ_q2)
    maxq = max(circ_q1, circ_q2)

    if not minq in offsets.keys():
        offsets[minq] = 0
    if not maxq in offsets.keys():
        offsets[maxq] = 0

    phys_idx1 = local_circuit_to_phys[circ_q1]
    phys_idx2 = local_circuit_to_phys[circ_q2]

    dist = abs(coupling_object.coupling_dist[phys_idx1][phys_idx2] - offsets[minq] - offsets[maxq])

    if dist > 1:
        offsets[minq] += dist // 2 - 1
        offsets[maxq] -= dist // 2 - 1

    return dist


def eval_cx_collection(cx_collection,
                       local_circuit_to_phys,
                       circ_qub_idx_limit,
                       coupling_object,
                       parameters,
                       plus_or_minus):
    """
    This is the heuristic cost function to evaluate the cost
    for a mapping configuration
    :param cx_collection: list of tuples representing CNOT qubits;
    control/target does not matter
    :param local_circuit_to_phys: permutation of circuit wires
    :param circ_qub_idx_limit: physical qubit index threshold to skip;
        gates that operate on qubits
        with index higher than limit are not considered
    :param attenuate: Boolean - if later changes in the layout
    should not affect the beginning of the circuit
    :return: evaluated cost of a mapping (ordering)
    """
    sum_eval = 0

    nr_ops_skipped = 0
    nr_ops_active = 0

    circ_qub_accumulated_offsets = {}

    """
    Effectively saying that this number of CNOTs was activated by increasing
    the qubit index to phys_qub_idx_limit
    """
    nr_ops_at_idx_limit = 0

    for q_tuple in cx_collection:

        if q_tuple[0] > circ_qub_idx_limit or q_tuple[1] > circ_qub_idx_limit:
            # increment the number of skipped gates from the collection
            nr_ops_skipped += 1
        else:
            nr_ops_active += 1

        if q_tuple[0] == circ_qub_idx_limit or q_tuple[1] == circ_qub_idx_limit:
            nr_ops_at_idx_limit += 1


    for cnot_index, q_tuple in enumerate(cx_collection):

        if q_tuple[0] > circ_qub_idx_limit or q_tuple[1] > circ_qub_idx_limit:
            continue

        # this is a linear distance between the qubits
        # it does not take into consideration the architecture
        """
            TODO: Different types of distances
        """
        # part1 = get_distance_linear(c1, c2)
        # part1 = get_distance_coupling(c1, c2, coupling_object)
        part1 = get_distance_offsets(q_tuple[0],
                                     q_tuple[1],
                                     circ_qub_accumulated_offsets,
                                     local_circuit_to_phys,
                                     coupling_object)

        if parameters["option_attenuate"]:
            # later changes in the layout should not affect
            # the beginning of the circuit

            # factor = len(cx_collection) / cnot_index
            # part1 *= factor

            # factor = 0.0001 + parameters["att_fact"] * (len(cx_collection) - cnot_index)/len(cx_collection)
            # part1 *= (1 - factor)
            # part1 *= math.log(1 + mult * factor, 2)
            # part1 *= math.log(mult * factor, 2)

            # part1 *=  math.exp(factor)
            # part1 *= math.exp(len(cx_collection) - cnot_index)

            # factor = cnot_index * cnot_index
            # part1 /= factor

            # part1 *= factor

            # Go Gaussian
            # x \in [0, 1]
            x = cnot_index/len(cx_collection)
            # b \in [0, 100]
            b = parameters["att_b"]
            # c \in [0, 1]
            c = parameters["att_c"]
            amplitude = math.exp(-b * (x - c) ** 2)
            part1 *= amplitude

        sum_eval += plus_or_minus * part1

    # if attenuate:
    #     # each additionally considered qubit should enable
    #     # as many CNOTs as possible
    #     # if not, penalise the cost function
    #     sum_eval += skipped * 200# * len(order)

    # print(limit, sum_eval)

    # print("check", qubit, "tmp sum", temp_cost, "order", order)
    if parameters["option_skipped_cnots"]:
        sk_factor = nr_ops_skipped #/ len(cx_collection)
        sk_factor *= parameters["penalty_skipped_cnot"]
        sum_eval += plus_or_minus * sk_factor

    """
        If the index increase did not add any additional CNOTs...math.inf cost?
    """
    if parameters["option_div_by_active"]:
        if nr_ops_at_idx_limit > 0:
            sum_eval /= nr_ops_at_idx_limit
        else:
            sum_eval = math.inf

    return sum_eval

'''
    Main
'''
def cuthill_order(dag_circuit, coupling_object, parameters):
    """
        Implementation below should be a kind of BFS similar to Cuthill

        https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm

        From Wikipedia:

        The Cuthill McKee algorithm is a variant of the standard breadth-first
        search algorithm used in graph algorithms. It starts with a peripheral
        node and then generates levels R i {\displaystyle R_{i}} R_{i}
        for i = 1 , 2 , . . {\displaystyle i=1,2,..} i=1, 2,..
        until all nodes are exhausted.

        The set R i + 1 {\displaystyle R_{i+1}} R_{i+1} is created
        from set R i {\displaystyle R_{i}} R_i by listing all vertices
        adjacent to all nodes in R i {\displaystyle R_{i}} R_{i}.
        These nodes are listed in increasing degree. This last detail is
        the only difference with the breadth-first search algorithm.

    :param dag_circuit:
    :param coupling_object:
    :return:
    """

    nr_nisq_qubits = parameters["nisq_qubits"]
    nr_circ_qubits = dag_circuit.num_qubits()

    # the start configuration is 0...nrq
    # order = list(range(nrq))

    #
    # Create a list of qubit index tuples representing the cnots
    # from the dag_circuit
    #
    # the cnot collection is initialised to empty
    cx_collection = []

    # use qiskit topological sort
    # nodes_collection = dag_circuit.topological_nodes()
    # each gate is transformed into a tuple of qubit indices
    for gate in dag_circuit.gate_nodes():
        # gate = dag_circuit.multi_graph.nodes[node]
        # gate = nodes_collection[node]

        if gate.name not in ["cx", "CX"]:
            continue

        q1 = int(gate.qargs[0].index)
        q2 = int(gate.qargs[1].index)

        cx_collection.append((q1, q2))

    # sume = eval_cx_collection(cx_collection, order, nrq)
    # print("naive start:", sume)

    '''
    A tree of nodes representing the qubits/wires of the circuit
    The idea is to find a permutation that minimises a given cost function
    '''
    options_tree = nx.DiGraph()
    # the id of the node in the tree
    maximum_nr_node = 0
    # add the first node to the tree
    # its name is ZERO and has cost zero
    # this is not correct actually, because the returned permutations will be
    # always starting with zero
    options_tree.add_node(maximum_nr_node, name=0, cost=0)
    maximum_nr_node += 1

    # order = [math.inf for x in order]
    curr_mapping = [math.inf] * nr_circ_qubits
    curr_mapping[0] = 0 #random.randint(0, nr_nisq_qubits - 1)

    # take each index at a time.
    # start from 1
    # the current limit is node_index
    for circ_qub_idx in range(1, nr_circ_qubits):

        # determine the leafs of the options_tree and store them in a list
        all_leafs = [x for x in options_tree.nodes() if options_tree.out_degree(x) == 0]

        '''
            Cut-Off search heuristic for placement
        '''
        if (circ_qub_idx % parameters["max_depth"] == 0) and (circ_qub_idx > 0):

            minnode, mincost = cuthill_evaluate_leafs(all_leafs, options_tree)

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
        # print("--------")
        for prev_leaf in all_leafs:
            # setup ordering based on parents of this node
            cuthill_set_partial_perm(circ_qub_idx, options_tree, curr_mapping, prev_leaf)
            # print(circ_qub_idx, curr_mapping)

            # Where to store candidates
            local_minimas = []

            # # is the processing past the limit?
            # # which is a ratio (e.g. 1/3) of the total number of qubits
            # # the idea being that initially, up to this index limit, the costs have to increase
            # # afterwards the costs have to decrease
            #
            reverse_cond = 1
            if parameters["option_max_then_min"]:
                if circ_qub_idx < nr_nisq_qubits / parameters["qubit_increase_factor"]:
                    # this condition is checked
                    # if the cost should be maximised -> towards start of permutation
                    reverse_cond = -1

            # the variable is used to store the evaluated cost
            hold_sum = reverse_cond * math.inf

            # This is the list of all NISQ qubits used for mapping by now
            leaf_ancestors = [options_tree.nodes[x]["name"]
                              for x in nx.ancestors(options_tree, prev_leaf)]
            leaf_ancestors.append(options_tree.nodes[prev_leaf]["name"])

            # print("ancestors", leaf_ancestors)

            # Use all the qubits which are not predecessors of the current leaf
            for phys_qubit in range(nr_nisq_qubits):

                if phys_qubit in leaf_ancestors:
                    # This nisq qubit has already been used
                    # print("skip ", phys_qubit)
                    continue
                # else:
                    # print("consider ", phys_qubit)

                #place previously unused qubit on index limit
                curr_mapping[circ_qub_idx] = phys_qubit

                # # the cost of the leaf is stored in prev_sum
                # prev_cost = options_tree.nodes[prev_leaf]["cost"]

                # evaluate the cost of the cnots touching the qubits before limit
                temp_cost = eval_cx_collection(cx_collection,
                                               curr_mapping,
                                               circ_qub_idx,
                                               coupling_object,
                                               parameters,
                                               reverse_cond)

                # the question is: is the computed cost less than the hold_sum?
                condition = (reverse_cond * temp_cost) < (reverse_cond * hold_sum)

                # if the condition is true
                # store the candidate node in the local_minimas
                # if temp_cost != prev_cost:
                if condition:
                    hold_sum = temp_cost
                    local_minimas.clear()
                    local_minimas.append(phys_qubit)
                elif temp_cost == hold_sum \
                        and len(local_minimas) < parameters["max_children"]:
                    local_minimas.append(phys_qubit)

                # reset placement
                # curr_mapping[circ_qub_idx] = math.inf

            # reset entire ordering
            # cuthill_set_partial_perm(math.inf, options_tree, curr_mapping, prev_leaf)


            # add leafs to the node that generated these permutations
            # print("add leafs ", local_minimas, " to the parent ", prev_leaf)
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

    # the final evaluation of the options_tree leafs
    all_leafs = [x for x in options_tree.nodes() if options_tree.out_degree(x) == 0]
    minnode, mincost = cuthill_evaluate_leafs(all_leafs, options_tree)

    # the final order permutation is computed.
    # this will be returned and used by the placement
    cuthill_set_partial_perm(nr_circ_qubits, options_tree, curr_mapping, minnode)

    # print("sum eval:", mincost, eval_cx_collection(cx_collection, order, nrq))
    # print(order)

    return curr_mapping

def cuthill_evaluate_leafs(all_leafs, options_tree):
    """
    Return leaf with minimum cost and its cost
    :param all_leafs: collection of leafs to analyse
    :param options_tree: tree where the leafs are from
    :return: leaf with minimum and its cost
    """
    minnode = -1
    mincost = math.inf

    for nd in all_leafs:
        if options_tree.nodes[nd]["cost"] < mincost:
            mincost = options_tree.nodes[nd]["cost"]
            minnode = nd

    if mincost == math.inf:
        # If no minimum was found, return the last leaf in the list
        minnode = all_leafs[-1]

    return minnode, mincost


def cuthill_set_partial_perm(circ_qubit_idx_limit, options_tree, curr_mapping, prev_leaf):
    """
    Construct the mapping backwards: from the highest circuit qubit
    index to the lowest

    :param circ_qubit_idx_limit:
    :param options_tree:
    :param curr_mapping:
    :param prev_leaf:
    :return:
    """

    # TODO: Adapt for multiple qubits on nisq
    circ_qubit = circ_qubit_idx_limit  - 1
    p_prev_leaf = prev_leaf

    while circ_qubit >= 0:
        # save the prev_leaf index
        phys_qubit = options_tree.nodes[p_prev_leaf]["name"]
        curr_mapping[circ_qubit] = phys_qubit

        circ_qubit -= 1
        if circ_qubit >= 0:
            p_prev_leaf = list(options_tree.pred[p_prev_leaf])[0]