import qiskit
from qiskit.transpiler import CouplingMap, PassManager, Layout
from qiskit.transpiler.passes import BasicSwap, SetLayout, ApplyLayout

import csv
from _private_benchmark.CONNECTION import INDEX_CONNECTION_LIST as connection_list
from ast import literal_eval

from k7m_core import K7MCompiler, K7MInitialMapping

gdv_name = "TFL"
depth_range = {
    "TFL" : [5 * x for x in range(1, 10)],
    "BSS" : [100 * x for x in range(1, 10)]
}
# gdv_name = "QSE"

qubits = {
    16 : "Aspen-4",
    20 : "Tokyo",
    53 : "Rochester",
    54 : "Sycamore"
}

nr_qubits = 16

first_run = True

def benchmark(depth, trail):

    # qasm_file_name = "_private_benchmark/BNTF/16QBT_{:02}CYC_{}_{}.qasm".format(
    #     depth, gdv_name, trail)
    #
    # solution_file_name = "_private_benchmark/meta/16QBT_{:02}CYC_{}_{}_solution.csv".format(
    #     depth, gdv_name, trail)

    depth_string = "{:03}".format(depth)
    folder = "BSS"
    if gdv_name == "TFL":
        folder = "BNTF"
        depth_string = "{:02}".format(depth)

    qasm_file_name = "_private_benchmark/{}/{}QBT_{}CYC_{}_{}.qasm".format(
        folder, nr_qubits, depth_string, gdv_name, trail)

    solution_file_name = "_private_benchmark/meta/{}QBT_{}CYC_{}_{}_solution.csv".format(
        nr_qubits, depth_string, gdv_name, trail)

    # print("qiskit", depth)
    # input qasm file as circuit
    test_circuit = qiskit.QuantumCircuit.from_qasm_file(qasm_file_name)
    """
        Construct the optimal initial mapping
    """
    qiskit_layout_dict = dict()
    original_nodes = list()
    with open(solution_file_name, 'r') as csvfile:
        for original_node in csv.reader(csvfile, delimiter=','):
            original_nodes.append(literal_eval(original_node[0]))
    csvfile.close()
    print(original_nodes)

    for i in range(len(original_nodes)):
        qiskit_layout_dict[test_circuit.qregs[0][i]] = original_nodes[i]
    # construct passes that use the optimal initial mapping and BasicSwap
    # however, no swapping gates should be ever necessary
    original_pm = PassManager()
    # print(original_nodes)
    qiskit_coupling_map = CouplingMap(couplinglist=connection_list[qubits[nr_qubits]])
    optimal_layout = Layout()
    optimal_layout.from_dict(qiskit_layout_dict)
    original_pm.append([SetLayout(optimal_layout),
                        ApplyLayout(),
                        BasicSwap(qiskit_coupling_map)])
    map_original_circuit = original_pm.run(test_circuit)
    optimal_depth = map_original_circuit.depth()
    print("optimal mapping: the circuit has", optimal_depth, "cycles")
    # print(map_original_circuit.draw(style="text"))
    # construct passes that use the DenseLayout+StochasticSwap without initial mapping


    """
       K7M 
    """
    gate_costs = {'id': 0, 'u1': 0, 'measure': 0,
                  'reset': 0, 'barrier': 0,
                  'u2': 1, 'u3': 1, 'U': 1,

                  'cx': 10, 'CX': 10,
                  "rev_cx_edge": 10,  # related to the edge in the coupling

                  "ok": 0,
                  # update the costs
                  "rev_cnot": 4 * 1 + 10,  # 4 u2/hadamard + 1 cnot
                  "swap": 3 * 10 + 4,  # 4 u2/hadamard + 3 cnot

                  'seed': 19}  # pass the seed through gate costs

    parameters = {
        # maximum depth of the search tree
        # after this depth, the leafs are evaluated
        # and only the path with minimum cost is kept in the tree
        # thus, the tree is pruned
        "max_depth": test_circuit.n_qubits,
        # "max_depth": test_circuit.n_qubits/4,

        # maximum number of children of a node
        # "max_children": qiskit_coupling_map.size(),
        "max_children": 3,

        # the first number_of_qubits * this factor the search maximises the cost
        # afterwards it minimises it
        "option_max_then_min": False,
        "qubit_increase_factor": 3,

        "option_skipped_cnots": False,
        "penalty_skipped_cnot": 200,

        "option_divide_by_activated" : False,

        # later changes in the mapping should not affect
        # the initial mapping of the circuit
        "option_attenuate": False,
    }

    parameters_string = str(parameters)

    # the number of qubits in the device
    parameters["nisq_qubits"] = qiskit_coupling_map.size()
    # Add the gate costs
    parameters["gate_costs"] = gate_costs
    # Should the initial mapping be chosen random?
    parameters["initial_map"]= K7MInitialMapping.HEURISTIC
    parameters["unidirectional_coupling"]=False
    parameters["dry_run"]= False

    k7mcomp = K7MCompiler(connection_list[qubits[nr_qubits]], parameters)
    map_test_circuit = k7mcomp.run(test_circuit)

    # print(map_test_circuit.draw(output="text", fold=-1))
    tmp_circuit = map_test_circuit.decompose()
    # print(tmp_circuit.draw(output="text", fold=-1))
    # tmp_circuit = qiskit_to_tk(map_test_circuit)
    # Transform.RebaseToQiskit().DecomposeSWAPtoCX().apply(tmp_circuit)
    depth_result = tmp_circuit.depth()

    print("k7m mapping: the circuit has", depth_result, "cycles")
    # print(map_test_circuit.draw(style="text"))
    # accumulate result
    print("----")

    file_op_type = "a"
    global first_run
    if first_run:
        file_op_type = "w"
    first_run = False

    with open(
            "_private_data/BNTF/{}_{}_{}.csv".format(gdv_name,
                                                     qubits[nr_qubits],
                                                  parameters_string
                                                     ),
            file_op_type) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([trail, "k7m", optimal_depth, depth_result])

    return optimal_depth, depth_result



for trail in range(10):
    for depth in depth_range[gdv_name]:
        optimal_depth, depth_result = benchmark(depth, trail)

