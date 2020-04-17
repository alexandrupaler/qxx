import copy
import numpy
import qiskit
import enum

from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import StochasticSwap, SetLayout, ApplyLayout, \
    Decompose, BasicSwap

from k7m_coupling import K7MCoupling
from k7m_positions import K7MPositions
from k7m_start_configuration import cuthill_order
import k7m_gate_utils as gs


class K7MInitialMapping(enum.Enum):
    RANDOM = enum.auto()
    LINEAR = enum.auto()
    HEURISTIC = enum.auto()


class K7MCompiler(TransformationPass):

    def __init__(self, coupling_map, parameters):

        self.parameters = parameters

        self.coupling_obj = K7MCoupling(coupling_map, parameters)

        self.positions_obj = None

        self.operation_costs = parameters["gate_costs"]


    def run(self, quantum_circuit):

        dag_circuit = circuit_to_dag(quantum_circuit)

        initial_mapping = []
        if self.parameters["initial_map"] == K7MInitialMapping.RANDOM:
            # Only the first positions which correspond to the circuit qubits
            initial_mapping = numpy.random.permutation(
                self.parameters["nisq_qubits"])
            initial_mapping = initial_mapping[:dag_circuit.num_qubits()]
        elif self.parameters["initial_map"] == K7MInitialMapping.LINEAR:
            initial_mapping = list(range(dag_circuit.num_qubits()))
        elif self.parameters["initial_map"] == K7MInitialMapping.HEURISTIC:
            initial_mapping = cuthill_order(dag_circuit, self.coupling_obj, self.parameters)


        # print(initial_mapping)
        #
        # return quantum_circuit
        print("                       .......")

        original_pm = PassManager()
        optimal_layout = Layout()
        for c_idx, p_idx in enumerate(initial_mapping):
            optimal_layout.add(quantum_circuit.qregs[0][c_idx], p_idx)

        original_pm.append([SetLayout(optimal_layout),
                            ApplyLayout(),
                            StochasticSwap(self.coupling_obj.coupling),
                            Decompose(gate=qiskit.extensions.SwapGate)])

        return original_pm.run(quantum_circuit)


        if self.positions_obj == None:
            self.positions_obj = K7MPositions(dag_circuit,
                                              self.parameters,
                                              initial_mapping)
        '''
            Start with an initial configuration
        '''
        compiled_dag, back_stack = self.find_solution(dag_circuit, self.parameters["dry_run"])

        """
            Returning here stops backtracking -> A full backtrack is not available,
            but the following code, after having iterated through the possible
            configurations (code before here):
            * counts the most common configuration
            * computes for each configuration the cost
            * chooses the configuration of minimum cost
        """

        # Clean the positions
        self.positions_obj = None

        return dag_to_circuit(compiled_dag)

        # name = compiled_dag.name or None
        # circuit = qiskit.QuantumCircuit(*compiled_dag.qregs.values(), *compiled_dag.cregs.values(), name=name)
        #
        # for node in compiled_dag.topological_op_nodes():
        #     # Get arguments for classical control (if any)
        #     inst = node.op.copy()
        #     inst.condition = node.condition
        #     circuit._append(inst, node.qargs, node.cargs)
        # return circuit

        # '''
        #     It is possible to collect all configurations encountered
        #     Slows everything, and no huge benefit on test circuits
        # '''
        # collect_config = {}
        # # analyse saved configurations
        # for el in back_stack:
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
        #     tmp_dag, back_stack = find_solution(coupling_map, coupling_object, collect_config[k][1],
        #                                                      current_who_at_index, dag_circuit, True)
        #
        #     # fourth element in the tuple is the cost
        #     tmp_solution_cost = sum(bs[3] for bs in back_stack)
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
        # compiled_dag, back_stack = find_solution(coupling_map, coupling_object, collect_config[min_current_positions][1],
        #                                             current_who_at_index, dag_circuit, False)
        #
        # return compiled_dag


    def save_first_swaps(self, dag_circuit):
        # print(coupling_map)
        # nodes_collection = nx.topological_sort(dag_circuit.multi_graph)
        # nodes_collection = dag_circuit.topological_nodes()
        # get first set of disjunct cnots
        # fcnots = first_set_of_disjunct_cnots(nodes_collection, dag_circuit.multi_graph, get_dag_nr_qubits(dag_circuit))
        fcnots = self.first_set_of_disjunct_cnots(dag_circuit, dag_circuit.num_qubits())

        for circuit_cnot in fcnots:
            phys_gate = self.positions_obj.translate_op_to_coupling_map(circuit_cnot)
            phys_qub1, phys_qub2 = get_cnot_qubits(phys_gate)

            start_phys1 = self.get_coupling_node_idx(phys_qub1)
            start_phys2 = self.get_coupling_node_idx(phys_qub2)

            stop_phys1, stop_phys2 = self.coupling_obj.heuristic_choose_coupling_edge(
                start_phys1,
                start_phys2
            )

            if start_phys1 != stop_phys1:
                self.move_qubit_from_to(start_phys1, stop_phys1)

            if start_phys2 != stop_phys2:
                self.move_qubit_from_to(start_phys2, stop_phys2)


    def find_solution(self, dag_circuit, dry_run):

        '''
        :param dag_circuit: input dag circuit
        :param dry_run: Execute without compiling a new circuit. For benchmarking purposes.
        :return:
        '''

        self.save_first_swaps(dag_circuit)

        # Simulate a stack
        back_stack = []

        '''
            Initialise the stack and the compiled solution
        '''
        compiled_dag = dag_circuit
        if not dry_run:
            compiled_dag = qiskit.dagcircuit.DAGCircuit()
            compiled_dag.add_qreg(self.positions_obj.quantum_reg)
            compiled_dag.add_creg(self.positions_obj.classic_reg)


        coupling_edge_idx = 0

        #make a list in order to allow forward iteration
        # nodes_collection = list(nx.topological_sort(dag_circuit.multi_graph))

        for original_op in dag_circuit.topological_op_nodes():
            # original_op = dag_circuit.multi_graph.node[gate]
            # if original_op["type"] not in ["op"]:
            #     continue

            # print(original_op)

            translated_op = self.positions_obj.translate_op_to_coupling_map(original_op)

            if translated_op.name not in ["cx", "CX"]:
                # print(op)
                '''
                    A place to include a heuristic
                '''
                if not dry_run:

                    gs.append_ops_to_dag(compiled_dag, [translated_op])

                    # if translated_op.name in ["u1", "u2", "u3"]:
                    #     # TODO: Not necessary. Was used here for speed purposes.
                    #     gs.append_ops_to_dag(compiled_dag, [translated_op])
                    #     # compiled_dag.apply_operation_back(translated_op.op, qargs=translated_op.qargs)
                    # else:
                    #     compiled_dag.apply_operation_back(translated_op.op,
                    #                                       qargs=translated_op.qargs,
                    #                                       cargs=translated_op.cargs)
            else:
                # continue
                '''
                    Found a CNOT:
                    Check that the qubits are on an edge
                '''
                qub1, qub2 = get_cnot_qubits(translated_op)
                gates_to_insert = []

                # How much does a movement cost?
                additional_cost = 0

                if self.coupling_obj.is_pair(qub1, qub2, self.parameters["unidirectional_coupling"]):
                    # can be directly implemented
                    gates_to_insert += gs.comp_cnot_gate_list(qub1, qub2,
                                                              self.positions_obj.quantum_reg,
                                                              inverse_cnot = False)
                    additional_cost = self.operation_costs["ok"]
                    # print("CNOT!!!", qub1, qub2, "from", get_cnot_qubits(original_op))

                elif self.coupling_obj.is_pair(qub2, qub1, self.parameters["unidirectional_coupling"]):
                    # needs a reversed cnot
                    gates_to_insert += gs.comp_cnot_gate_list(qub2, qub1,
                                                              self.positions_obj.quantum_reg,
                                                              inverse_cnot = True)
                    additional_cost = self.operation_costs["rev_cnot"]
                    # print("CNOT!!!", qub2, qub1, "from", get_cnot_qubits(original_op))

                else:
                    # print("do not add this", qub1, qub2)
                    '''
                        qub 1 and qub2 are not a coupling_map edge
                        Compute a solution
                    '''
                    start_phys_q_1 = self.get_coupling_node_idx(qub1)
                    start_phys_q_2 = self.get_coupling_node_idx(qub2)

                    '''
                        A Look-ahead/behind to see which edges were/will 
                        be used and to go towards them
                        Not used
                    '''
                    #get the next cnots and check use their coordinates to find the next edge
                    next_nodes = []
                    # self.commented_method_for_lookahead(next_nodes)

                    # for succ in dag_circuit.quantum_successors(original_op):
                    #     next_nodes.append(
                    #         self.positions_obj.pos_circuit_to_phys[succ.qar])
                    #     next_nodes.append(
                    #         self.positions_obj.pos_circuit_to_phys[succ.qar])

                    # Compute the edgee index where the qubits could be moved
                    stop_phys_q1, stop_phys_q2 = self.coupling_obj.heuristic_choose_coupling_edge(
                        start_phys_q_1,
                        start_phys_q_2,
                        next_nodes)

                    # #Determine the indices of the edge nodes where the qubits will be moved
                    # stop_node_index1 = coupling_object.coupling_edges_list[coupling_edge_idx][0]
                    # stop_node_index2 = coupling_object.coupling_edges_list[coupling_edge_idx][1]

                    if start_phys_q_1 == stop_phys_q2 and start_phys_q_2 == stop_phys_q1:
                        # the qubits sit on opposite positions than expected
                        # do not compute routes
                        # but make an inverse cnot
                        if not dry_run:
                            gates_to_insert += gs.comp_cnot_gate_list(start_phys_q_2, start_phys_q_1, self.positions_obj.quantum_reg, inverse_cnot = True)

                    else:
                        if start_phys_q_1 != stop_phys_q1:
                            # move the first qubit to the first edge node
                            # route1, \
                            route1q = self.move_qubit_from_to(start_phys_q_1, stop_phys_q1)

                            # a circuit is not generated, do not place gates
                            ret_gates_to_insert, part_cost = self.compute_swap_chain_and_cost(route1q, dry_run)
                            additional_cost += part_cost
                            if not dry_run:
                                gates_to_insert += ret_gates_to_insert

                            '''
                                Update: the previous swaps may have moved this qubit around
                            '''
                            translated_op = self.positions_obj.translate_op_to_coupling_map(original_op)
                            qub1, qub2 = get_cnot_qubits(translated_op)
                            start_phys_q_2 = self.get_coupling_node_idx(qub2)

                        if start_phys_q_2 != stop_phys_q2:
                            #move the second qubit to the second edge node
                            # route2, \
                            route2q = self.move_qubit_from_to(start_phys_q_2, stop_phys_q2)
                            # a circuit is not generated, do not place gates
                            ret_gates_to_insert, part_cost = self.compute_swap_chain_and_cost(route2q, dry_run)
                            additional_cost += part_cost
                            if not dry_run:
                                gates_to_insert += ret_gates_to_insert

                            '''
                                Update: the previous swaps may have moved qub1 backwards
                            '''
                            # before refactoring
                            # if stop_phys_q1 in route2:
                            # after refactoring
                            if stop_phys_q1 in route2q:
                                # qubit_node_index1 = get_coupling_node_idx(qub1, coupling, current_positions)

                                # The qubit before stop_qubit...Why am I writing it like this?
                                # before refactoring
                                # start_phys_q_1 = route2[route2.index(stop_phys_q1) - 1]
                                # after refactoring
                                start_phys_q_1 = route2q[
                                    route2q.index(stop_phys_q1) - 1]

                                # this if-statement seems useless
                                if start_phys_q_1 != stop_phys_q1:
                                    # route1, \
                                    route1q = self.move_qubit_from_to(start_phys_q_1, stop_phys_q1)

                                    ret_gates_to_insert, part_cost = self.compute_swap_chain_and_cost(route1q, dry_run)
                                    additional_cost += part_cost
                                    if not dry_run:
                                        gates_to_insert += ret_gates_to_insert

                    '''
                        It should be possible to implement the CNOT now
                    '''
                    # retranslate
                    retranslated_op = self.positions_obj.translate_op_to_coupling_map(original_op)
                    qub1, qub2 = get_cnot_qubits(retranslated_op)

                    if self.coupling_obj.is_pair(qub1, qub2, self.parameters["unidirectional_coupling"]):
                        if not dry_run:
                            gates_to_insert += gs.comp_cnot_gate_list(qub1, qub2,
                                                                      self.positions_obj.quantum_reg,
                                               inverse_cnot = False)
                        additional_cost += self.operation_costs["ok"]
                        # print("CNOT!!!", qub1, qub2, "from", get_cnot_qubits(original_op))

                    elif self.coupling_obj.is_pair(qub2, qub1, self.parameters["unidirectional_coupling"]):
                        if not dry_run:
                            gates_to_insert += gs.comp_cnot_gate_list(qub2, qub1,
                                                                      self.positions_obj.quantum_reg,
                                                                      inverse_cnot = True)
                        additional_cost += self.operation_costs["rev_cnot"]
                        # print("CNOT!!!", qub2, qub1, "from", get_cnot_qubits(original_op))

                if not dry_run:
                    gs.append_ops_to_dag(compiled_dag, gates_to_insert)

                # the others are not deep copied
                back_stack.append((copy.deepcopy(self.positions_obj),
                                           original_op,# gate,
                                           coupling_edge_idx,
                                           additional_cost)
                                          )

        return compiled_dag, back_stack


    def get_coupling_node_idx(self, qubit):
        # return coupling.qubits[("q", config_which_index[qubit])]
        # TODO: FIX THIS!!!
        # return coupling_object.coupling.qubits[("q", qubit)]
        return qubit


    def compute_swap_chain_and_cost(self, route, dry_run):
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

                """
                If the coupling map does not include qub1,qub2
                then check for qub2,qub1 and perform swap between indices
                if the latter does not exist either --> ERROR
                """
                if not self.coupling_obj.is_pair(qub1, qub2, self.parameters["unidirectional_coupling"]):
                    qub1, qub2 = qub2, qub1  # swap variables
                    if not self.coupling_obj.is_pair(qub1, qub2, self.parameters["unidirectional_coupling"]):
                        print("NOT GOOD: Coupling not OK!", qub1, qub2)
                        is_error = True

                if not is_error:
                    # print("swap", qub1, qub2)
                    """
                    This works correct because the 
                    qub1, qub2 variables are swapped above
                    The CNOTs are always according to the correct coupling graph
                    """
                    route_gate_list += gs.comp_cnot_gate_list(qub1, qub2,
                                                              self.positions_obj.quantum_reg,
                                                              inverse_cnot=False)

                    if self.parameters["unidirectional_coupling"]:
                        route_gate_list += gs.comp_cnot_gate_list(qub1, qub2,
                                                                  self.positions_obj.quantum_reg,
                                                                  inverse_cnot=True)
                    else:
                        route_gate_list += gs.comp_cnot_gate_list(qub2, qub1,
                                                                  self.positions_obj.quantum_reg,
                                                                  inverse_cnot=False)
                    route_gate_list += gs.comp_cnot_gate_list(qub1, qub2,
                                                              self.positions_obj.quantum_reg,
                                                              inverse_cnot=False)

                qub1 = qtmp

        # print("-------")
        route_cost = (len(route) - 1) * self.operation_costs["swap"]

        return route_gate_list, route_cost


    def commented_method_for_lookahead(self, nodes_collection, gate, dag_circuit):
        """
            The code was where the method is commented
            It has to do with the clustering of the CNOTs used in 
            coupling_object.heuristic_choose_coupling_edge_idx()

            Its translation is not finished
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
        #     t_ni_op = self.positions_obj.translate_op_to_coupling_map(ni_op)
        #
        #     ni_q1, ni_q2 = get_cnot_qubits(t_ni_op)
        #     ni_q1_i = self.get_coupling_node_idx(ni_q1)
        #     ni_q2_i = self.get_coupling_node_idx(ni_q2)
        #
        #     if ni_q1_i not in next_nodes and ni_q1_i not in [qubit_node_index1, qubit_node_index2]:
        #         next_nodes.append(ni_q1_i)
        #     if ni_q2_i not in next_nodes and ni_q2_i not in [qubit_node_index1, qubit_node_index2]:
        #         next_nodes.append(ni_q2_i)
        pass


    def first_set_of_disjunct_cnots(self, dag_circuit, maxqubits):
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


    def move_qubit_from_to(self, start_phys, stop_phys):

        '''
            Move the first qubit to the first edge position
        '''
        # route1 = coupling_object.reconstruct_route(start_phys, stop_phys)
        route_phys = self.coupling_obj.reconstruct_route(start_phys, stop_phys)
        # TODO: FIX IT!!!!
        # route1q = [coupling_object.coupling.index_to_qubit[r][1] for r in route1]
        # Get the qubits from the circuit...not their indices
        # route_phys = [positions_object.pos_phys_to_circuit[r] for r in route1]

        '''
            Update the maps
        '''
        # route1log = [current_who_at_index[x] for x in route1q]
        # positions_object.update_configuration(route1log)
        self.positions_obj.update_configuration(route_phys)
        # positions_object.debug_configuration()

        # return route1, route_phys
        return route_phys


def get_cnot_qubits(op):
    qubit1 = op.qargs[0].index

    qubit2 = -1
    if len(op.qargs) == 2:
        qubit2 = op.qargs[1].index

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