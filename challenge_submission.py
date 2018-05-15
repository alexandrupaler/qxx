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

h_data = {
    "opaque": False,
    "n_args": 0,
    "n_bits": 1,
    "args": [],
    "bits": ["a"],
    # gate h a { u2(0,pi) a; }
    "body": node.GateBody([
        node.CustomUnitary([
            node.Id("u2", 0, ""),
            node.ExpressionList([
                node.Int(0),
                node.Real(sympy.pi)
            ]),
            node.PrimaryList([
                node.Id("a", 0, "")
            ])
        ])
    ])
}

def construct_coupling_weights(coupling_map):
    weights = []

    return weights


def add_reverse_edges_and_weights(coupling, gatecosts):
    # get all edges from coupling
    edg = copy.deepcopy(coupling.G.edges)
    for edg in edg:
        coupling.G.remove_edge(*edg)

        # the direct edge gets a weight
        coupling.G.add_edge(edg[0], edg[1], weight=gatecosts["cx"])

        # the inverse edge
        # three CNOTS + four Hadamards for the swap
        coupling.G.add_edge(edg[1], edg[0], weight=3*gatecosts["cx"] + 4*gatecosts["u2"])


def choose_initial_configuration(dag_circuit):
    """
        Returns an int-to-int map: logical qubit @ physical qubit map
        There is something similar in Coupling, but I am not using it
    """
    configuration = {}

    for i in range(dag_circuit.num_cbits()):
        configuration[i] = i # qubit i@i

    return configuration

#def update_configuration(route, configuration):
    


def get_position_of_qubits(qub1, qub2, configuration):
    return (configuration[qub1], configuration[qub2])


def is_configuration_ok(qub1, qub2, configuration, coupling):
    pos1, pos2 = get_position_of_qubits(qub1, qub2, configuration)

    is_ok = False
    is_ok = is_ok or ((pos1, pos2) in coupling.edges())
    is_ok = is_ok or ((pos2, pos1) in coupling.edges())

    return is_ok


def get_next_coupling_edge(backtracking_stack, coupling_edges_list):
    coupling_edge_index = backtracking_stack[-1][3]
    if coupling_edge_index + 1 < len(coupling_edges_list):
        return coupling_edges_list[coupling_edge_index + 1]
    #no edge remaining
    return None


#def get_swaps_for_edge(the_edge, coupling, coupling_dist):





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

    # How do you find out the number of qubits in the architecture if coupling_map is None?
    #if coupling_map is None:
    #    #create all to all graph

    '''
        Prepare the Floyd Warshall graph and weight matrix
        Coupling related objects
    '''
    coupling = Coupling(coupling_map)
    add_reverse_edges_and_weights(coupling, gate_costs)
    coupling_pred, coupling_dist = nx.floyd_warshall_predecessor_and_distance(coupling.G, weight="weight")
    coupling_edges_list = [ e for e in coupling.G.edges()]
    print(coupling.G.edges().data())
    print(coupling_pred)
    print(coupling_dist)


    '''
        Current logical qubit position on physical qubits
    '''
    current_positions = choose_initial_configuration(dag_circuit)
    print(current_positions)

    #portile in ordinea lor din circuit

    #incearca sa pui un hadamard



    all_gates = nx.topological_sort(dag_circuit.multi_graph)
    for gate in all_gates:
        nd = dag_circuit.multi_graph.node[gate]
        if nd["type"] == "op" and (nd["name"] not in ["measure", "barrier"]):
            print(nd)

    '''
        Simulate a stack
    '''
    backtracking_stack = []

    '''
        Initialise the stack
    '''
    coupling_edge_idx = 0
    backtracking_stack.append( (copy.deepcopy(current_positions), gate, coupling_edge_idx))

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



    compiled_dag = dag_circuit
    return compiled_dag


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
     compiled_dag = direction_mapper(compiled_dag, coupling)
    # # # # Simplify cx gates
    # # cx_cancellation(compiled_dag)
    # # # # Simplify single qubit gates
     compiled_dag = optimize_1q_gates(compiled_dag)
    #
    # #####################
    # # Put your code here
    # #####################
    #
    # # Return the compiled dag circuit