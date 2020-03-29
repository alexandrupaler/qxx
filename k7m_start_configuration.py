import networkx as nx
import math


def get_distance_coupling(c1, c2, coupling_object):
    """

    :param c1:
    :param c2:
    :param coupling_object:
    :return:
    """

    idx1 = coupling_object.coupling.qubits[("q", c1)]
    idx2 = coupling_object.coupling.qubits[("q", c2)]

    dist = coupling_object.coupling_dist[idx1][idx2]

    return dist


def get_distance_linear(c1, c2):
    """

    :param c1:
    :param c2:
    :return:
    """
    return abs(c1 - c2)


def get_distance_offsets(c1, c2, offsets, coupling_object):

    # dist = abs(offsets[c1] + c1 - c2 - offsets[c2])

    # idx1 = coupling_object["coupling"].qubits[("q", c1)]
    # idx2 = coupling_object["coupling"].qubits[("q", c2)]

    idx1 = coupling_object.coupling.physical_qubits[c1]
    idx2 = coupling_object.coupling.physical_qubits[c2]

    minq = min(c1, c2)
    maxq = max(c1, c2)

    dist = abs(coupling_object.coupling_dist[idx1][idx2] - offsets[minq] - offsets[maxq])

    if dist > 1:
        offsets[minq] += dist // 2 - 1
        offsets[maxq] -= dist // 2 - 1

    return dist


def eval_cx_collection(cx_collection,
                       order,
                       limit,
                       coupling_object,
                       attenuate=False):
    """
    This is the heuristic cost function to evaluate the cost
    for a mapping configuration
    :param cx_collection: list of tuples representing CNOT qubits;
    control/target does not matter
    :param order: permutation of circuit wires
    :param limit: qubit index threshold to skip; gates that operate on qubits
        with index higher than limit are not considered
    :param attenuate: Boolean - if later changes in the layout
    should not affect the beginning of the circuit
    :return: evaluated cost of a mapping (ordering)
    """
    sum_eval = 0

    t = 0
    skipped = 0
    active = 0
    thisturnactive = 0

    for tu in cx_collection:
        c1 = order[tu[0]]
        c2 = order[tu[1]]

        if c1 > limit or c2 > limit:
            # increment the number of skipped gates from the collection
            skipped += 1
        else:
            active += 1

            if c1 == limit or c2 == limit:
                thisturnactive += 1

    offsets = []
    for qi in range(limit + 1):
        offsets.append(0)

    for tu in cx_collection:
        c1 = order[tu[0]]
        c2 = order[tu[1]]

        # increment the index counter of the cnot gate
        t += 1

        if c1 > limit or c2 > limit:
            continue

        # this is a linear distance between the qubits
        # it does not take into consideration the architecture
        """
            TODO: Different types of distances
        """
        # part1 = get_distance_linear(c1, c2)
        # part1 = get_distance_coupling(c1, c2, coupling_object)
        part1 = get_distance_offsets(c1, c2, offsets, coupling_object)

        if attenuate:
            # later changes in the layout should not affect
            # the beginning of the circuit

            # factor = len(cx_collection) / t
            # part1 *= factor

            mult = 7
            factor = (len(cx_collection) - t)/len(cx_collection)
            # part1 *= (1 - factor)
            part1 *= math.log(1 + mult * factor, 2)

            # factor = t * t
            # part1 /= factor

            # part1 *= factor

        sum_eval += part1

    # if attenuate:
    #     # each additionally considered qubit should enable
    #     # as many CNOTs as possible
    #     # if not, penalise the cost function
    #     sum_eval += skipped * 200# * len(order)

    # print(limit, sum_eval)

    if thisturnactive > 0:
        sum_eval /= thisturnactive
    else:
        sum_eval = math.inf

    return sum_eval, skipped

'''
    Main
'''
def cuthill_order(dag_circuit, coupling_object, parameters):
    """
        Need to check what the implementation below does, but it should be
        a kind of BFS similar to Cuthill ordering

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

    nrq = dag_circuit.width()

    # the start configuration is 0...nrq
    order = list(range(nrq))

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

        # if gate["name"] not in ["cx"]:
        if gate.name not in ["cx"]:
            continue

        # q1 = int(gate["qargs"][0][1])
        # q2 = int(gate["qargs"][1][1])
        #
        q1 = int(gate.qargs[0][1])
        q2 = int(gate.qargs[1][1])

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

    order = [math.inf for x in order]

    # take each index at a time.
    # start from 1
    for node_index in range(1, nrq):

        # the current limit is node_index
        limit = node_index

        # determine the leafs of the options_tree and store them in a list
        all_leafs = [x for x in options_tree.nodes() if options_tree.out_degree(x) == 0]

        '''
            Cut-Off search heuristic for placement
        '''
        if limit % parameters["max_depth"] == 0:

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

        for prev_leaf in all_leafs:
            # setup ordering based on parents of this node
            cuthill_set_partial_perm(limit, options_tree, order, prev_leaf)

            #where to store candidates
            local_minimas = []

            # the variable is used to store the evaluated cost
            hold_sum = math.inf

            # 31.08.2018
            # # is the processing past the limit?
            # # which is a ratio (e.g. 1/3) of the total number of qubits
            # # the idea being that initially, up to this index limit, the costs have to increase
            # # afterwards the costs have to decrease
            # if limit < nrq / parameter_qubit_increase_factor:
            #     hold_sum = - math.inf

            leaf_ancestors = [options_tree.nodes[x]["name"] for x in nx.ancestors(options_tree, prev_leaf)]
            leaf_ancestors.append(options_tree.nodes[prev_leaf]["name"])

            # Use all the qubits which are not predecessors of the current leaf
            for qubit in range(nrq):

                if qubit in leaf_ancestors:
                    continue

                #place previously unused qubit on index limit
                order[qubit] = limit

                # the cost of the leaf is stored in prev_sum
                prev_sum = options_tree.nodes[prev_leaf]["cost"]

                # evaluate the cost of the cnots touching the qubits before limit
                sume, skipped = eval_cx_collection(cx_collection,
                                                   order, limit,
                                                   coupling_object,
                                                   attenuate = True)

                # print("check", qubit, "tmp sum", sume, "order", order)

                # the question is: is the computed cost less than the hold_sum?
                # this condition is checked if the cost should be minimised -> towards end of permutation

                # 31.08.2018
                # if limit < nrq / parameter_qubit_increase_factor:
                #     # this condition is checked if the cost should be maximised -> towards start of permutation
                #     sume -= (skipped * parameter_skipped_cnot_penalty)
                #     condition = sume > hold_sum
                # else:
                #     sume += (skipped * parameter_skipped_cnot_penalty)
                #     condition = sume < hold_sum

                # sume += (skipped * parameter_skipped_cnot_penalty)
                condition = (sume < hold_sum)

                # if the condition is true
                # store the candidate node in the local_minimas
                if sume != prev_sum:
                    if condition:
                        hold_sum = sume
                        local_minimas.clear()
                        local_minimas.append(qubit)
                    elif sume == hold_sum and len(local_minimas) < parameters["max_children"]:
                        local_minimas.append(qubit)

                # reset placement
                order[qubit] = math.inf

            #reset entire ordering
            cuthill_set_partial_perm(math.inf, options_tree, order, prev_leaf)

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

    # the final evaluation of the options_tree leafs
    all_leafs = [x for x in options_tree.nodes() if options_tree.out_degree(x) == 0]
    minnode, mincost = cuthill_evaluate_leafs(all_leafs, options_tree)

    # the final order permutation is computed.
    # this will be returned and used by the placement
    cuthill_set_partial_perm(nrq, options_tree, order, minnode)

    # print("sum eval:", mincost, eval_cx_collection(cx_collection, order, nrq))
    # print(order)

    return order

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


def cuthill_set_partial_perm(limit, options_tree, order, prev_leaf):
    """

    :param limit:
    :param options_tree:
    :param order:
    :param prev_leaf:
    :return:
    """
    position = math.inf
    if limit != math.inf:
        position = limit - 1

    # this node is placed at the index position
    order[options_tree.nodes[prev_leaf]["name"]] = position

    # save the prev_leaf index
    p_prev_leaf = prev_leaf
    # the parents of prev_leaf and so on towards the root are placed
    # at indices decremented from position
    while len(options_tree.pred[p_prev_leaf]) == 1:
        # get the parent of the current node,
        # store it into a list and take the first element
        p_prev_leaf = list(options_tree.pred[p_prev_leaf])[0]
        if limit != math.inf:
            # decrement position
            position -= 1
        # place current node at the lower position
        order[options_tree.nodes[p_prev_leaf]["name"]] = position


