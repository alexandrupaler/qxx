# -*- coding: utf-8 -*-

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
---------------> please fill out this section <---------------

Your Name :

Your E-Mail :

Description of the algorithm :

- How does the algorithm work?
- Did you use any previously published schemes? Cite relevant papers.
- What packages did you use in your code and for what part of the algorithm?
- How general is your approach? Does it work for arbitrary coupling layouts (qubit number)?
- Are there known situations when the algorithm fails?


---------------> please fill out this section <---------------
"""

# Include any Python modules needed for your implementation here
import networkx as nx
import copy

from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper import swap_mapper, direction_mapper, cx_cancellation, optimize_1q_gates, Coupling
from qiskit import qasm, unroll

from gatesignatures import add_signatures_to_circuit, get_unrolled_qasm

def add_reverse_edges_and_weights(coupling, gatecosts):
    # get all edges from coupling
    edgs = copy.deepcopy(coupling.G.edges)
    for edg in edgs:
        # print(edg)
        coupling.G.remove_edge(*edg)

        # the direct edge gets a weight
        coupling.G.add_edge(edg[0], edg[1], weight=gatecosts["cx"])

        # the inverse edge
        # three CNOTS + four Hadamards for the swap
        coupling.G.add_edge(edg[1], edg[0], weight=3*gatecosts["cx"] + 4*gatecosts["u2"])


def get_dag_nr_qubits(dag_circuit):
    """
    Get the number of qubits of the circuit
    Assume the circuit has a single qubit register called "q"
    """
    nrq = dag_circuit.qregs["q"]
    return nrq


def choose_initial_configuration(dag_circuit):
    """
        Returns an int-to-int map: logical qubit @ physical qubit map
        There is something similar in Coupling, but I am not using it
    """
    configuration = {}

    for i in range(get_dag_nr_qubits(dag_circuit)):
        configuration[i] = i # qubit i@i

    return configuration

def get_position_of_qubits(qub1, qub2, configuration):
    return configuration[qub1], configuration[qub2]

#
# def is_configuration_ok(qub1, qub2, configuration, coupling):
#     pos1, pos2 = get_position_of_qubits(qub1, qub2, configuration)
#
#     is_ok = False
#     is_ok = is_ok or ((pos1, pos2) in coupling.edges())
#     is_ok = is_ok or ((pos2, pos1) in coupling.edges())
#
#     return is_ok


def get_next_coupling_edge(backtracking_stack, coupling_edges_list):
    """
        Reads the stack and advances the index of the edge to send the qubits to
        Not used until now, because the backtracking step is not implemented
    """
    coupling_edge_index = backtracking_stack[-1][3]
    if coupling_edge_index + 1 < len(coupling_edges_list):
        return coupling_edges_list[coupling_edge_index + 1]
    #no edge remaining
    return None


def reconstruct_route(start, stop, coupling_pred):
    route = []
    if coupling_pred[start][stop] is None:
        # can/should this happen?
        return []
    else:
        route.append(stop)
        while start != stop:
            stop = coupling_pred[start][stop]
            route.append(stop)

    return list(reversed(route))


def move_qubits_to_edge(qub1, qub2, coupling_edge, coupling, coupling_pred):

    # print(coupling_edge)

    qub1_to_index = coupling.qubits[("q", qub1)]
    qub2_to_index = coupling.qubits[("q", qub2)]

    route1 = []
    route2 = []

    if qub1_to_index == coupling_edge[1] and qub2_to_index == coupling_edge[0]:
        #The qubits are ok placed on the edge, and need a single SWAP
        route1 = reconstruct_route(qub1_to_index, coupling_edge[0], coupling_pred)
    else:
        if qub1_to_index != coupling_edge[0]:
            # only in the case I do need to move
            route1 = reconstruct_route(qub1_to_index, coupling_edge[0], coupling_pred)

        if qub2_to_index != coupling_edge[1]:
            #only in the case I do need to move
            route2 = reconstruct_route(qub2_to_index, coupling_edge[1], coupling_pred)

            if qub2_to_index in route1:
                '''
                    The same problem as below
                '''
                prevq2 = route1[route1.index(qub2_to_index) - 1]
                if prevq2 == route2[1]:
                    route2 = route2[2:]
                else:
                    route2 = [prevq2] + route2

    if len(route1) > 0 and (route1[-1] in route2):
        '''
            This is a problem: along route2 the destination of route1 will be moved
            It has to be put back
            Take the predecesor of route1[-1] from route2 and append it to route1
        '''
        prevq = route2[route2.index(route1[-1]) - 1]
        if route1[-2] == prevq:
            route1.pop()
            route1.pop()
        else:
            route1.append(prevq)

    #translate graph node ids to physical qubits
    #take only their ints, and leave the "q" out
    routeq1 = [coupling.index_to_qubit[r][1] for r in route1]
    routeq2 = [coupling.index_to_qubit[r][1] for r in route2]

    return routeq1, routeq2

def is_pair_in_coupling_map(qubit1, qubit2, coupling_map):
    pair_pass = (qubit1 in coupling_map)
    if pair_pass:
        pair_pass &= qubit2 in coupling_map[qubit1]

    return pair_pass


def update_configuration(route, configuration):
    initial = []

    if len(route) == 0:
        return

    nr = copy.deepcopy(route)

    for i in range(0, len(nr)):
        initial.append(configuration[nr[i]])

    nr.append(nr[0])
    nr = nr[1:]

    for i in range(0, len(nr)):
        configuration[nr[i]] = initial[i]


def compute_swap_chain(route, coupling_map):
    """
        Returns a list of tuple of the form
        (g, ['q', i])? where g is "h" or "cx"
    """
    #print(coupling_map)
    ret = []

    if len(route) == 0:
        return ret

    qub1 = route[0]
    for qub2 in route[1:]:
        qtmp = qub2
        is_error = False
        # is_reversed = True
        if not is_pair_in_coupling_map(qub1, qub2, coupling_map):
            qub1, qub2 = qub2, qub1 #swap variables
            # is_reversed = True
            if not is_pair_in_coupling_map(qub1, qub2, coupling_map):
                print("NOT GOOD: Coupling not OK!", qub1, qub2)
                is_error = True

        if not is_error:
            #print(qub1, qub2)

            ret += compute_cnot_gate_list(qub1, qub2, False)
            ret += compute_cnot_gate_list(qub1, qub2, True)
            ret += compute_cnot_gate_list(qub1, qub2, False)

            # ret.append(("cx", [("q", qub1), ("q", qub2)]))
            # ret.append(("h", [("q", qub1)]))
            # ret.append(("h", [("q", qub2)]))
            # ret.append(("cx", [("q", qub1), ("q", qub2)]))
            # ret.append(("h", [("q", qub1)]))
            # ret.append(("h", [("q", qub2)]))
            # ret.append(("cx", [("q", qub1), ("q", qub2)]))

        qub1 = qtmp

    return ret


def compute_cnot_gate_list(qub1, qub2, inverse_cnot):
    ret = []

    if not inverse_cnot:
        ret.append({"name": "cx", "qargs": [("q", qub1), ("q", qub2)]})
    else:
        ret.append({"name": "h", "qargs": [("q", qub1)]})
        ret.append({"name": "h", "qargs": [("q", qub2)]})
        ret.append({"name": "cx", "qargs": [("q", qub1), ("q", qub2)]})
        ret.append({"name": "h", "qargs": [("q", qub1)]})
        ret.append({"name": "h", "qargs": [("q", qub2)]})

    # if is_pair_in_coupling_map(qub1, qub2, coupling_map):
    #     ret.append("cx", [("q", qub1), ("q", qub2)])
    # elif is_pair_in_coupling_map(qub2, qub1, coupling_map):
    #     ret.append(("h", [("q", qub1)]))
    #     ret.append(("h", [("q", qub2)]))
    #     ret.append("cx", [("q", qub1), ("q", qub2)])
    #     ret.append(("h", [("q", qub1)]))
    #     ret.append(("h", [("q", qub2)]))

    return ret


def append_ops_to_dag(dag_circuit, op_list):

    for op in op_list:
        dag_circuit.apply_operation_back(op["name"], op["qargs"])#, op["cargs"], op["params"], op["condition"])

    return dag_circuit.node_counter


def compiler_function(dag_circuit, coupling_map=None, gate_costs=None):
    """
    Modify a DAGCircuit based on a gate cost function.

    Instructions:
        Your submission involves filling in the implementation
        of this function. The function takes as input a DAGCircuit
        object, which can be generated from a QASM file by using the
        function 'qasm_to_dag_circuit' from the included 
        'submission_evaluation.py' module. For more information
        on the DAGCircuit object see the or QISKit documentation
        (eg. 'help(DAGCircuit)').

    Args:
        dag_circuit (DAGCircuit): DAGCircuit object to be compiled.
        coupling_circuit (list): Coupling map for device topology.
                                 A coupling map of None corresponds an
                                 all-to-all connected topology.
        gate_costs (dict) : dictionary of gate names and costs.

    Returns:
        A modified DAGCircuit object that satisfies an input coupling_map
        and has as low a gate_cost as possible.
    """

    #swaps = compute_swap_chain ([e for e in range(10)], coupling_map)
    #print(swaps)

    #current_positions = choose_initial_configuration(dag_circuit)
    #update_configuration([15, 0], current_positions)

    '''
        Prepare the Floyd Warshall graph and weight matrix
        Coupling related objects
    '''
    # How do you find out the number of qubits in the architecture if coupling_map is None?
    coupling = Coupling(coupling_map)
    add_reverse_edges_and_weights(coupling, gate_costs)
    coupling_pred, coupling_dist = nx.floyd_warshall_predecessor_and_distance(coupling.G, weight="weight")
    coupling_edges_list = [ e for e in coupling.G.edges()]

    # print(coupling_map)
    # # print(coupling_pred)
    #
    # r1, r2 = move_qubits_to_edge(1, 0, coupling_edges_list[5], coupling, coupling_pred)
    # print(r1)
    # print(r2)
    #
    # return

    # print(coupling.G.edges().data())
    # print(coupling_pred)
    # print(coupling_dist)
    # print("----------")
    # print(coupling_map)

    #
    #
    # print(dag_circuit.node_counter)
    # dag_circuit.apply_operation_back("h", [("q", 1)])
    # print(dag_circuit.node_counter)
    #
    # print("-------")

    '''
        Simulate a stack
    '''
    backtracking_stack = []

    '''
        Initialise the stack and the compiled solution
    '''
    dag_circuit = add_signatures_to_circuit(dag_circuit)
    compiled_dag = clone_dag_without_gates(dag_circuit)
    coupling_edge_idx = 0
    last_gate_idx = compiled_dag.node_counter

    '''
        Current logical qubit position on physical qubits
    '''
    current_positions = choose_initial_configuration(dag_circuit)

    #backtracking_stack.append((copy.deepcopy(current_positions), gate AICI NU STIU CE VINE, coupling_edge_idx, last_gate_idx))


    '''
        Try to build a solution
    '''
    nodes_collection = nx.topological_sort(dag_circuit.multi_graph)
    for gate in nodes_collection:
        op_orig = dag_circuit.multi_graph.node[gate]
        if op_orig["type"] not in ["op"]:
            continue

        op = translate_op_to_coupling_map(op_orig, current_positions)

        if op["name"] not in ["cx", "CX"]:
            #print(op)
            '''
                A place to include a heuristic
            '''
            compiled_dag.apply_operation_back(op["name"], op["qargs"], op["cargs"], op["params"], op["condition"])
        else:
            '''
                Found a CNOT:
                Check that the qubits are on an edge
            '''
            qub1, qub2 = get_cnot_qubits(op)

            gates_to_insert = []

            if is_pair_in_coupling_map(qub1, qub2, coupling_map):
                gates_to_insert += compute_cnot_gate_list(qub1, qub2, False)
                print("CNOT!!!", qub1, qub2, "from", get_cnot_qubits(op_orig))
            elif is_pair_in_coupling_map(qub2, qub1, coupling_map):
                gates_to_insert += compute_cnot_gate_list(qub2, qub1, True)
                print("CNOT!!!", qub2, qub1, "from", get_cnot_qubits(op_orig))
            else:
                print("do not add this", qub1, qub2)
                '''
                    qub 1 and qub2 are not a coupling_map edge
                    Compute a solution
                '''
                coupling_edge_idx = 0 #put heuristic here to select edge

                route1, route2 = move_qubits_to_edge(qub1, qub2,
                                                     coupling_edges_list[coupling_edge_idx],
                                                     coupling, coupling_pred)

                gates_to_insert += compute_swap_chain(route1, coupling_map)
                gates_to_insert += compute_swap_chain(route2, coupling_map)

                update_configuration(route1, current_positions)
                update_configuration(route2, current_positions)

                #retranslate
                op = translate_op_to_coupling_map(op_orig, current_positions)
                qub1, qub2 = get_cnot_qubits(op)
                if is_pair_in_coupling_map(qub1, qub2, coupling_map):
                    gates_to_insert += compute_cnot_gate_list(qub1, qub2, False)
                    print("CNOT!!!", qub1, qub2, "from", get_cnot_qubits(op_orig))
                elif is_pair_in_coupling_map(qub2, qub1, coupling_map):
                    gates_to_insert += compute_cnot_gate_list(qub2, qub1, True)
                    print("CNOT!!!", qub2, qub1, "from", get_cnot_qubits(op_orig))

            current_gate_count = append_ops_to_dag(compiled_dag, gates_to_insert)


    print("--------- Check ------------")
    tmp_solution = get_unrolled_qasm(compiled_dag)
    tmp_solution_cost, mapped_ok = check_solution_and_compute_cost(tmp_solution, coupling_map, gate_costs)

    print(tmp_solution_cost, mapped_ok)
    if mapped_ok:
        print("Seems OK")



    return tmp_solution


    # ##layout
    # layout_max_index = max(map(lambda x: x[1] + 1, layout.values()))
    # # Circuit for this swap slice
    # circ = DAGCircuit()
    # circ.add_qreg('q', layout_max_index)
    # circ.add_basis_element("CX", 2)
    # circ.add_basis_element("cx", 2)
    # circ.add_basis_element("swap", 2)
    # circ.add_gate_data("cx", cx_data)
    # circ.add_gate_data("swap", swap_data)



 # # # look into the dag_circuit
    # # # and generate a coupling graph based on it?
    # # cxNodes = dag_circuit.get_named_nodes("cx")
    # # for node in cxNodes:
    # #     args = dag_circuit.multi_graph.nodes[node]["qargs"]
    # #     control = args[0][1]
    # #     target = args[1][1]
    # #     print(str(control) + " " + str(target))
    #
    #
    #
    # #
    # # Construieste un graf orientat din coupling map
    # #
    #
    # #
    # # Scrie o functie care plaseaza SWAP intr-un circuit
    # #
    #
    #
    # # #####################
    # # # Put your code here
    # # #####################
    #
    # initial_layout = None
    # compiled_dag, final_layout = swap_mapper(copy.deepcopy(dag_circuit),
    #                                          coupling, initial_layout,
    #                                          trials=40, seed=19)
    #
    # #
    # # Post processing?
    # #
    # # Expand swaps
    # # compiled_dag = dag_circuit
    # basis_gates = "u1,u2,u3,cx,id"  # QE target basis
    # program_node_circuit = qasm.Qasm(data=compiled_dag.qasm()).parse()
    # unroller_circuit = unroll.Unroller(program_node_circuit,
    #                                    unroll.DAGBackend(
    #                                        basis_gates.split(",")))
    # compiled_dag = unroller_circuit.execute()
    #
    # # Change cx directions
    # compiled_dag = direction_mapper(compiled_dag, coupling)
    # # # # Simplify cx gates
    # # cx_cancellation(compiled_dag)
    # # # # Simplify single qubit gates
    # compiled_dag = optimize_1q_gates(compiled_dag)
    #
    # #####################
    # # Put your code here
    # #####################
    #
    # # Return the compiled dag circuit

def translate_op_to_coupling_map(op, current_positions):
    '''
        Translate qargs and cargs depending on the position of the logical qubit
    '''

    retop = {}
    retop["name"] = op["name"]
    retop["type"] = op["type"]
    retop["params"] = op["params"]
    retop["condition"] = op["condition"]


    retop["qargs"] = []
    for qa in op["qargs"]:
        retop["qargs"].append(("q", current_positions[qa[1]]))

    retop["cargs"] = []
    for ca in op["cargs"]:
        retop["cargs"].append(("c", current_positions[ca[1]]))

    return retop


def clone_dag_without_gates(dag_circuit):
    compiled_dag = DAGCircuit()
    compiled_dag.basis = copy.deepcopy(dag_circuit.basis)
    compiled_dag.gates = copy.deepcopy(dag_circuit.gates)
    compiled_dag.add_qreg("q", get_dag_nr_qubits(dag_circuit))
    compiled_dag.add_creg("c", get_dag_nr_qubits(dag_circuit))
    return compiled_dag


def check_solution_and_compute_cost(dag_circuit, coupling_map, gate_costs):
    coupling_map_passes = True
    cost = 0

    #get nodes after topological sort
    nodescollection = nx.topological_sort(dag_circuit.multi_graph)
    for gate in nodescollection:
        op = dag_circuit.multi_graph.node[gate]

        if op["name"] not in gate_costs:
            continue

        cost += gate_costs.get(op["name"])  # compute cost
        if op["name"] in ["cx", "CX"] \
                and coupling_map is not None:  # check coupling map

            qubit1, qubit2 = get_cnot_qubits(op)

            pair_pass = is_pair_in_coupling_map(qubit1, qubit2, coupling_map)

            if not pair_pass:
                print("here", qubit1, qubit2)

            coupling_map_passes &= pair_pass

    return cost, coupling_map_passes


def get_cnot_qubits(op):
    qubit1 = op["qargs"][0][1]
    qubit2 = op["qargs"][1][1]
    return qubit1, qubit2


def debug_configuration(current_positions):
    for i in range(0, len(current_positions)):
        print("q", i, "@", current_positions[i])

