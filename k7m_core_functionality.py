# Include any Python modules needed for your implementation here
import networkx as nx
import math
import sympy
import collections
import numpy

# from qiskit.mapper import swap_mapper, direction_mapper, cx_cancellation, \
#     optimize_1q_gates, Coupling
# from qiskit import qasm, unroll

from gatesignatures import add_signatures_to_circuit
from gatesimplifiers import paler_cx_cancellation, paler_simplify_1q

import copy

# from qiskit.mapper import swap_mapper, direction_mapper,
# cx_cancellation, optimize_1q_gates, Coupling
# from qiskit import qasm, unroll

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.dagcircuit import DAGCircuit

from k7m_coupling_object import CouplingObject
from k7m_positions_object import PositionsObject

global_operation_costs = {"swap": 34, "rev_cnot": 4, "ok": 0}

# def get_position_of_qubits(qub1, qub2, configuration):
#     return configuration[qub1], configuration[qub2]


# def get_next_coupling_edge(backtracking_stack, coupling_edges_list):
#     """
#         Reads the stack and advances the index of the edge to send the qubits to
#         Not used until now, because the backtracking step is not implemented
#     """
#     coupling_edge_index = backtracking_stack[-1][3]
#     if coupling_edge_index + 1 < len(coupling_edges_list):
#         return coupling_edges_list[coupling_edge_index + 1]
#     #no edge remaining
#     return None

def get_coupling_node_idx(qubit, coupling_object, config_which_index):
    # return coupling.qubits[("q", config_which_index[qubit])]
    # TODO: FIX THIS!!!
    # return coupling_object.coupling.qubits[("q", qubit)]
    return qubit

# def reconstruct_route(start, stop, coupling_pred):
#     # route = []
#     route = collections.deque()
#
#     if coupling_pred[start][stop] is None:
#         # can/should this happen?
#         return []
#     else:
#         # route.append(stop)
#         route.appendleft(stop)
#         while start != stop:
#             stop = coupling_pred[start][stop]
#             # route.append(stop)
#             route.appendleft(stop)
#
#     # return list(reversed(route))
#     return route

# def update_configuration(route, configuration, current_who_at_index):
#     initial = []
#
#     if len(route) == 0:
#         return
#
#     nr = copy.deepcopy(route)
#
#     for i in range(0, len(nr)):
#         initial.append(configuration[nr[i]])
#
#     nr.append(nr[0])
#     nr = nr[1:]
#
#     for i in range(0, len(nr)):
#         configuration[nr[i]] = initial[i]
#
#     current_who_at_index.update({v: k for k, v in configuration.items()})


def compute_swap_chain_and_cost(route, coupling_obj, dry_run):
    """
        Returns a list of tuple of the form
        (g, ['q', i])? where g is "h" or "cx"
    """
    route_gate_list = []

    if len(route) == 0:
        return [], 0

    # print("route phys qubits", route)

    if not dry_run:
        qub1 = route[0]
        for qub2 in route[1:]:
            qtmp = qub2
            is_error = False
            if not coupling_obj.is_pair(qub1, qub2):
                qub1, qub2 = qub2, qub1 #swap variables
                if not coupling_obj.is_pair(qub1, qub2):
                    print("NOT GOOD: Coupling not OK!", qub1, qub2)
                    is_error = True

            if not is_error:
                # print("swap", qub1, qub2)
                route_gate_list += compute_cnot_gate_list(qub1, qub2, inverse_cnot = False)
                route_gate_list += compute_cnot_gate_list(qub1, qub2, inverse_cnot = True)
                route_gate_list += compute_cnot_gate_list(qub1, qub2, inverse_cnot = False)

            qub1 = qtmp

    # print("-------")
    route_cost = (len(route) - 1) * global_operation_costs["swap"]

    return route_gate_list, route_cost


def compute_cnot_gate_list(qub1, qub2, inverse_cnot = False):
    ret = []

    if not inverse_cnot:
        ret.append({"name": "cx", "qargs": [("q", qub1), ("q", qub2)], "params": None})
    else:
        ret.append({"name": "u2", "qargs": [("q", qub1)], "params": [sympy.N(0), sympy.N(sympy.pi)]})
        ret.append({"name": "u2", "qargs": [("q", qub2)], "params": [sympy.N(0), sympy.N(sympy.pi)]})

        ret.append({"name": "cx", "qargs": [("q", qub1), ("q", qub2)], "params": None})

        ret.append({"name": "u2", "qargs": [("q", qub1)], "params": [sympy.N(0), sympy.N(sympy.pi)]})
        ret.append({"name": "u2", "qargs": [("q", qub2)], "params": [sympy.N(0), sympy.N(sympy.pi)]})

    return ret

def append_ops_to_dag(dag_circuit, op_list):

    for op in op_list:
        if paler_cx_cancellation(dag_circuit, op):
            if paler_simplify_1q(dag_circuit, op):
                dag_circuit.apply_operation_back(op["name"],
                                                 op["qargs"],
                                                 params=op["params"])

    return dag_circuit.node_counter


# def qubit_graph(dag_circuit, negativeNodes=False):
#
#     qgraph = nx.DiGraph()
#
#     nodes_collection = nx.topological_sort(dag_circuit.multi_graph)
#
#     edges_coll = {}
#     for node in nodes_collection:
#         gate = dag_circuit.multi_graph.nodes[node]
#         if gate["name"] in ["cx", "CX"]:
#             # found a cnot
#             qub1, qub2 = get_cnot_qubits(gate)
#             edg = (qub1, qub2)
#             if negativeNodes:
#                 edg = (-qub1, -qub2)#used for spectral_layout
#             if edg not in edges_coll:
#                 edges_coll[edg] = 0
#             edges_coll[edg] += 1
#
#     for edg in edges_coll:
#         qgraph.add_edge(edg[0], edg[1], weight=edges_coll[edg])
#
#     return qgraph


def k7m_compiler_function(dag_circuit,
                          coupling_map = None,
                          gate_costs = None):

    # update the costs
    global_operation_costs["rev_cnot"] = 4 * gate_costs["u2"]
    global_operation_costs["swap"] = 3 * gate_costs["cx"] + 4

    coupling_obj = CouplingObject(coupling_map, gate_costs)

    positions_obj = PositionsObject(dag_circuit, coupling_obj, random = False)

    '''
        Start with an initial configuration
    '''
    compiled_dag, backtracking_stack = find_solution(coupling_obj,
                                                     positions_obj,
                                                     dag_circuit,
                                                     dry_run= False)

    """
        Returning here stops backtracking -> A full backtrack is not available,
        but the following code, after having iterated through the possible
        configurations (code before here):
        * counts the most common configuration
        * computes for each configuration the cost
        * chooses the configuration of minimum cost
    """

    return compiled_dag

    # '''
    #     It is possible to collect all configurations encountered
    #     Slows everything, and no huge benefit on test circuits
    # '''
    # collect_config = {}
    # # analyse saved configurations
    # for el in backtracking_stack:
    #     kkk = hash(frozenset(el[0].items()))
    #     if kkk not in collect_config:
    #         collect_config[kkk] = (0, el[0])
    #     collect_config[kkk] = (collect_config[kkk][0] + 1, collect_config[kkk][1])
    #
    #
    # '''
    #     Find the configuration with the smallest cost
    # '''
    # min_cost = math.inf
    # min_current_positions = "what is here?"
    # for k in collect_config:
    #     # print(collect_config[k][0], ":", collect_config[k][1])
    #     current_who_at_index = {v: k for k, v in collect_config[k][1].items()}
    #     tmp_dag, backtracking_stack = find_solution(coupling_map, coupling_object, collect_config[k][1],
    #                                                      current_who_at_index, dag_circuit, True)
    #
    #     # fourth element in the tuple is the cost
    #     tmp_solution_cost = sum(bs[3] for bs in backtracking_stack)
    #
    #     # tmp_solution_cost, mapped_ok = check_solution_and_compute_cost(tmp_dag, coupling_map, gate_costs)
    #     if tmp_solution_cost <= min_cost:
    #         # compiled_dag = tmp_dag
    #         min_cost = tmp_solution_cost
    #         min_current_positions = k
    #     # mapped_ok = "did not check"
    #     # print(tmp_solution_cost, mapped_ok)
    #
    # '''
    #     The minimum cost configuration will be used
    # '''
    # current_who_at_index = {v: k for k, v in collect_config[min_current_positions][1].items()}
    #
    # compiled_dag, backtracking_stack = find_solution(coupling_map, coupling_object, collect_config[min_current_positions][1],
    #                                             current_who_at_index, dag_circuit, False)
    #
    # return compiled_dag


def save_first_swaps(coupling_object, positions_object, dag_circuit):
    # print(coupling_map)
    # nodes_collection = nx.topological_sort(dag_circuit.multi_graph)
    # nodes_collection = dag_circuit.topological_nodes()
    # get first set of disjunct cnots
    # fcnots = first_set_of_disjunct_cnots(nodes_collection, dag_circuit.multi_graph, get_dag_nr_qubits(dag_circuit))
    fcnots = first_set_of_disjunct_cnots(dag_circuit, dag_circuit.num_qubits())

    for circuit_cnot in fcnots:
        phys_gate = positions_object.translate_op_to_coupling_map(circuit_cnot)
        phys_qub1, phys_qub2 = get_cnot_qubits(phys_gate)

        start_phys1 = get_coupling_node_idx(phys_qub1, coupling_object, positions_object)
        start_phys2 = get_coupling_node_idx(phys_qub2, coupling_object, positions_object)

        stop_phys1, stop_phys2 = coupling_object.heuristic_choose_coupling_edge(
            start_phys1,
            start_phys2
        )

        # stop_node_index1 = coupling_object.coupling_edges_list[coupling_edge_idx][0]
        # stop_node_index2 = coupling_object.coupling_edges_list[coupling_edge_idx][1]

        if start_phys1 != stop_phys1:
            move_qubit_from_to(start_phys1, stop_phys1,
                               coupling_object, positions_object)

        if start_phys2 != stop_phys2:
            move_qubit_from_to(start_phys2, stop_phys2,
                               coupling_object, positions_object)


def find_solution(
        coupling_object,
        positions_object,
        dag_circuit,
        dry_run
):

    '''

    :param coupling_map:
    :param coupling_object:
    :param current_positions:
    :param current_who_at_index:
    :param dag_circuit: input dag circuit
    :param dry_run: Execute without compiling a new circuit. For benchmarking purposes.
    :return:
    '''

    save_first_swaps(coupling_object,
                     positions_object,
                     dag_circuit)

    # Simulate a stack
    backtracking_stack = []

    '''
        Initialise the stack and the compiled solution
    '''
    dag_circuit = add_signatures_to_circuit(dag_circuit)
    compiled_dag = None
    if not dry_run:
        compiled_dag = clone_dag_without_gates(dag_circuit)
    else:
        compiled_dag = dag_circuit

    coupling_edge_idx = 0

    #make a list in order to allow forward iteration
    # nodes_collection = list(nx.topological_sort(dag_circuit.multi_graph))

    for original_op in dag_circuit.topological_op_nodes():
        # original_op = dag_circuit.multi_graph.node[gate]
        # if original_op["type"] not in ["op"]:
        #     continue

        # print(original_op)

        translated_op = positions_object.translate_op_to_coupling_map(original_op)

        if translated_op.name not in ["cx", "CX"]:
            # print(op)
            '''
                A place to include a heuristic
            '''
            if not dry_run:
                if translated_op.name in ["u1", "u2", "u3"]:
                    # TODO: Not necessary. Was used here for speed purposes.
                    append_ops_to_dag(compiled_dag, [translated_op])
                else:
                    compiled_dag.apply_operation_back(translated_op["name"],
                                                      translated_op["qargs"],
                                                      translated_op["cargs"],
                                                      translated_op["params"],
                                                      translated_op["condition"])
        else:
            '''
                Found a CNOT:
                Check that the qubits are on an edge
            '''
            qub1, qub2 = get_cnot_qubits(translated_op)
            gates_to_insert = []

            # How much does a movement cost?
            additional_cost = 0

            if coupling_object.is_pair(qub1, qub2):
                # can be directly implemented
                gates_to_insert += compute_cnot_gate_list(qub1, qub2, inverse_cnot = False)
                additional_cost = global_operation_costs["ok"]
                # print("CNOT!!!", qub1, qub2, "from", get_cnot_qubits(original_op))

            elif coupling_object.is_pair(qub2, qub1):
                # needs a reversed cnot
                gates_to_insert += compute_cnot_gate_list(qub2, qub1, inverse_cnot = True)
                additional_cost = global_operation_costs["rev_cnot"]
                # print("CNOT!!!", qub2, qub1, "from", get_cnot_qubits(original_op))

            else:
                # print("do not add this", qub1, qub2)
                '''
                    qub 1 and qub2 are not a coupling_map edge
                    Compute a solution
                '''
                start_node_index1 = get_coupling_node_idx(qub1, coupling_object, positions_object)
                start_node_index2 = get_coupling_node_idx(qub2, coupling_object, positions_object)

                '''
                    A Look-ahead/behind to see which edges were/will 
                    be used and to go towards them
                    Not used
                '''
                #get the next cnots and check use their coordinates to find the next edge
                next_nodes = []
                commented_method_for_lookahead()

                # Compute the edgee index where the qubits could be moved
                stop_node_index1, stop_node_index2 = coupling_object.heuristic_choose_coupling_edge(
                    start_node_index1,
                    start_node_index2,
                    next_nodes)

                # #Determine the indices of the edge nodes where the qubits will be moved
                # stop_node_index1 = coupling_object.coupling_edges_list[coupling_edge_idx][0]
                # stop_node_index2 = coupling_object.coupling_edges_list[coupling_edge_idx][1]

                if start_node_index1 == stop_node_index2 and start_node_index2 == stop_node_index1:
                    # the qubits sit on opposite positions than expected
                    # do not compute routes
                    # but make an inverse cnot
                    if not dry_run:
                        gates_to_insert += compute_cnot_gate_list(start_node_index2, start_node_index1, inverse_cnot = True)

                else:
                    if start_node_index1 != stop_node_index1:
                        # move the first qubit to the first edge node
                        route1, route1q = move_qubit_from_to(start_node_index1, stop_node_index1,
                                                                         coupling_object,
                                                                         positions_object)

                        # a circuit is not generated, do not place gates
                        ret_gates_to_insert, part_cost = compute_swap_chain_and_cost(route1q, coupling_object, dry_run)
                        additional_cost += part_cost
                        if not dry_run:
                            gates_to_insert += ret_gates_to_insert

                        '''
                            Update: the previous swaps may have moved this qubit around
                        '''
                        translated_op = positions_object.translate_op_to_coupling_map(original_op)
                        qub1, qub2 = get_cnot_qubits(translated_op)
                        start_node_index2 = get_coupling_node_idx(qub2, coupling_object, positions_object)

                    if start_node_index2 != stop_node_index2:
                        #move the second qubit to the second edge node
                        route2, route2q = move_qubit_from_to(start_node_index2, stop_node_index2,
                                                                         coupling_object,
                                                                         positions_object)
                        # a circuit is not generated, do not place gates
                        ret_gates_to_insert, part_cost = compute_swap_chain_and_cost(route2q, coupling_object, dry_run)
                        additional_cost += part_cost
                        if not dry_run:
                            gates_to_insert += ret_gates_to_insert

                        '''
                            Update: the previous swaps may have moved qub1 backwards
                        '''
                        if stop_node_index1 in route2:
                            # qubit_node_index1 = get_coupling_node_idx(qub1, coupling, current_positions)

                            start_node_index1 = route2[route2.index(stop_node_index1) - 1]

                            # this if-statement seems useless
                            if start_node_index1 != stop_node_index1:
                                route1, route1q = move_qubit_from_to(start_node_index1, stop_node_index1,
                                                                                 coupling_object,
                                                                                 positions_object)

                                ret_gates_to_insert, part_cost = compute_swap_chain_and_cost(route1q, coupling_object, dry_run)
                                additional_cost += part_cost
                                if not dry_run:
                                    gates_to_insert += ret_gates_to_insert

                '''
                    It should be possible to implement the CNOT now
                '''
                # retranslate
                retranslated_op = positions_object.translate_op_to_coupling_map(original_op)
                qub1, qub2 = get_cnot_qubits(retranslated_op)

                if coupling_object.is_pair(qub1, qub2):
                    if not dry_run:
                        gates_to_insert += compute_cnot_gate_list(qub1, qub2, inverse_cnot = False)
                    additional_cost += global_operation_costs["ok"]
                    # print("CNOT!!!", qub1, qub2, "from", get_cnot_qubits(original_op))
                elif coupling_object.is_pair(qub2, qub1):
                    if not dry_run:
                        gates_to_insert += compute_cnot_gate_list(qub2, qub1, inverse_cnot = True)
                    additional_cost += global_operation_costs["rev_cnot"]
                    # print("CNOT!!!", qub2, qub1, "from", get_cnot_qubits(original_op))

            if not dry_run:
                append_ops_to_dag(compiled_dag, gates_to_insert)

            # the others are not deep copied
            backtracking_stack.append((copy.deepcopy(current_positions),
                                       original_op,# gate,
                                       coupling_edge_idx,
                                       additional_cost)
                                      )

    return compiled_dag, backtracking_stack


def commented_method_for_lookahead():
    pass
    """
        The code was where the method is commented
        It has to do with the clustering of the CNOTs used in 
        coupling_object.heuristic_choose_coupling_edge_idx()
    """
    # ni_index = nodes_collection.index(gate)
    # #use 3 cnots
    # for ni in range(0):
    #     ni_index -= 1
    #     if ni_index == -1:  # len(nodes_collection):
    #         break
    #
    #     ni_id = nodes_collection[ni_index]
    #     ni_op = dag_circuit.multi_graph.node[ni_id]
    #     while ni_op["name"] not in ["cx", "CX"]:
    #         ni_index -= 1
    #         if ni_index == 0:#len(nodes_collection):
    #             break
    #         ni_id = nodes_collection[ni_index]
    #         ni_op = dag_circuit.multi_graph.node[ni_id]
    #
    #     if ni_index == 0:#len(nodes_collection):
    #         break
    #
    #     t_ni_op = translate_op_to_coupling_map(ni_op, current_positions)
    #
    #     ni_q1, ni_q2 = get_cnot_qubits(t_ni_op)
    #     ni_q1_i = get_coupling_node_idx(ni_q1, coupling_object["coupling"], current_positions)
    #     ni_q2_i = get_coupling_node_idx(ni_q2, coupling_object["coupling"], current_positions)
    #     if ni_q1_i not in next_nodes and ni_q1_i not in [qubit_node_index1, qubit_node_index2]:
    #         next_nodes.append(ni_q1_i)
    #     if ni_q2_i not in next_nodes and ni_q2_i not in [qubit_node_index1, qubit_node_index2]:
    #         next_nodes.append(ni_q2_i)


def first_set_of_disjunct_cnots(dag_circuit, maxqubits):
    used_qubits = []
    return_cnots = []

    nodes_collection = dag_circuit.gate_nodes()

    for gate in nodes_collection:
        # gate = graph.nodes[node]
        if gate.name in ["cx", "CX"]:
            #found a cnot

            qub1, qub2 = get_cnot_qubits(gate)
            if (qub1 not in used_qubits) and (qub2 not in used_qubits):
                used_qubits.append(qub1)
                used_qubits.append(qub2)

                return_cnots.append(gate)

        if len(used_qubits) == maxqubits:
            """
                All possible qubits are already used. Stop CNOT search.
            """
            break

    return return_cnots


def move_qubit_from_to(start_phys,
                       stop_phys,
                       coupling_object,
                       positions_object,
                       ):

    '''
        Move the first qubit to the first edge position
    '''
    # route1 = coupling_object.reconstruct_route(start_phys, stop_phys)
    route_phys = coupling_object.reconstruct_route(start_phys, stop_phys)
    # TODO: FIX IT!!!!
    # route1q = [coupling_object.coupling.index_to_qubit[r][1] for r in route1]
    # Get the qubits from the circuit...not their indices
    # route_phys = [positions_object.pos_phys_to_circuit[r] for r in route1]

    '''
        Update the maps
    '''
    # route1log = [current_who_at_index[x] for x in route1q]
    # positions_object.update_configuration(route1log)
    positions_object.update_configuration(route_phys)
    # positions_object.debug_configuration()

    # return route1, route_phys
    return route_phys


def clone_dag_without_gates(dag_circuit):
    compiled_dag = DAGCircuit()
    compiled_dag.basis = copy.deepcopy(dag_circuit.basis)
    compiled_dag.gates = copy.deepcopy(dag_circuit.gates)
    # compiled_dag.add_qreg(QuantumRegister(get_dag_nr_qubits(dag_circuit), "q"))
    # compiled_dag.add_creg(ClassicalRegister(get_dag_nr_qubits(dag_circuit), "c"))

    compiled_dag.add_qreg(QuantumRegister(dag_circuit.num_qubits(), "q"))
    compiled_dag.add_creg(ClassicalRegister(dag_circuit.num_clbits(), "c"))

    return compiled_dag


def get_cnot_qubits(op):
    qubit1 = op.qargs[0][1]

    qubit2 = -1
    if len(op.qargs) == 2:
        qubit2 = op.qargs[1][1]

    return qubit1, qubit2

# def check_solution_and_compute_cost(dag_circuit, coupling_map, gate_costs):
#     coupling_map_passes = True
#     cost = 0
#
#     #get nodes after topological sort
#     nodescollection = nx.topological_sort(dag_circuit.multi_graph)
#     for gate in nodescollection:
#         op = dag_circuit.multi_graph.node[gate]
#
#         # print(op)
#
#         if op["name"] not in gate_costs:
#             continue
#
#         cost += gate_costs.get(op["name"])  # compute cost
#         if op["name"] in ["cx", "CX"] \
#                 and coupling_map is not None:  # check coupling map
#
#             qubit1, qubit2 = get_cnot_qubits(op)
#
#             pair_pass = is_pair_in_coupling_map(qubit1, qubit2, coupling_map)
#
#             if not pair_pass:
#                 print("here", qubit1, qubit2)
#
#             coupling_map_passes &= pair_pass
#
#     return cost, coupling_map_passes

# def compiler_function_nlayout(dag_circuit, coupling_map=None, gate_costs=None):
#     """
#     Modify a DAGCircuit based on a gate cost function.
#
#     Instructions:
#         Your submission involves filling in the implementation
#         of this function. The function takes as input a DAGCircuit
#         object, which can be generated from a QASM file by using the
#         function 'qasm_to_dag_circuit' from the included
#         'submission_evaluation.py' module. For more information
#         on the DAGCircuit object see the or QISKit documentation
#         (eg. 'help(DAGCircuit)').
#
#     Args:
#         dag_circuit (DAGCircuit): DAGCircuit object to be compiled.
#         coupling_circuit (list): Coupling map for device topology.
#                                  A coupling map of None corresponds an
#                                  all-to-all connected topology.
#         gate_costs (dict) : dictionary of gate names and costs.
#
#     Returns:
#         A modified DAGCircuit object that satisfies an input coupling_map
#         and has as low a gate_cost as possible.
#     """
#
#     initial_layout = {}
#
#     # coupling = Coupling(coupling_map)
#     coupling = CouplingMap(coupling_map)
#
#     coupling_object = {"coupling": CouplingMap(coupling_map)}
#     add_reverse_edges_and_weights_one(coupling_object["coupling"])
#     coupling_object["coupling_pred"], coupling_object["coupling_dist"] = nx.floyd_warshall_predecessor_and_distance(
#         coupling_object["coupling"].G, weight="weight")
#     coupling_object["coupling_edges_list"] = [e for e in coupling_object["coupling"].G.edges()]
#
#     """
#         Compute an initial mapping of the qubits
#     """
#     from startconfiguration import cuthill_order
#     y = cuthill_order(dag_circuit, coupling_object)
#     for i in range(len(y)):
#         initial_layout[('q', i)] = ('q', y[i])  # qubit i@i
#
#     print(initial_layout)
#     # end paler 31.08.2018
#
#     compiled_dag, final_layout = swap_mapper(copy.deepcopy(dag_circuit),
#                                              coupling, initial_layout,
#                                              trials=40, seed=gate_costs['seed'])
#
#     # Expand swaps
#     basis_gates = "u1,u2,u3,cx,id"  # QE target basis
#     program_node_circuit = qasm.Qasm(data=compiled_dag.qasm()).parse()
#     unroller_circuit = unroll.Unroller(program_node_circuit,
#                                        unroll.DAGBackend(
#                                            basis_gates.split(",")))
#     compiled_dag = unroller_circuit.execute()
#     # Change cx directions
#     compiled_dag = direction_mapper(compiled_dag, coupling)
#     # Simplify cx gates
#     compiled_dag = CXCancellation().run(compiled_dag)
#     # Simplify single qubit gates
#     compiled_dag = Optimize1qGates().run(compiled_dag)
#     # Return the compiled dag circuit
#     return compiled_dag