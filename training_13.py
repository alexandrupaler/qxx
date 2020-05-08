import qiskit
from qiskit.transpiler import CouplingMap, PassManager, Layout
from qiskit.transpiler.passes import BasicSwap, SetLayout, ApplyLayout

import csv
from _private_benchmark.CONNECTION import INDEX_CONNECTION_LIST as connection_list
from ast import literal_eval

from k7m_core import K7MCompiler, K7MInitialMapping

import time

from qiskit.converters import circuit_to_dag
import networkx as nx

gdv_name = "TFL"
depth_range = {
    "TFL" : [5 * x for x in range(1, 10)],
    "QSE" : [100 * x for x in range(1, 10)]
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

def convert_to_weighted_graph(multigraph):
    G = nx.Graph()
    for u,v in multigraph.edges(data=False):
        w = 1
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)

    return G


def dfs(i, visited, goesTo):
    # If it is already visited
    if (visited[i] == 1):
        return 0

    visited[i] = 1
    x = dfs(goesTo[i], visited, goesTo)

    return (x + 1)


def noOfTranspositions(P1, P, n):
    visited = [0] * n

    # This array stores which element goes to which position
    goesTo = [0] * n
    # building the goesTo[] array
    for i in range(n):
        goesTo[P[i]] = P1[i]

    # Initializing visited[] array
    for i in range(0, n):
        visited[i] = 0

    transpositions = 0

    for i in range(0, n):
        if (visited[i] == 0):
            ans = dfs(i, visited, goesTo)
            transpositions += ans - 1

    return transpositions

def circuit_analysis(depth, trail):
    if gdv_name == "TFL":
        folder = "BNTF"
        depth_string = "{:02}".format(depth)
        name_end = "TFL"
        if nr_qubits == 54:
            name_end = "QSE"

    elif gdv_name == "QSE":
        folder = "BSS"
        depth_string = "{:03}".format(depth)
        name_end = "QSE"

    # if nr_qubits==54:
    #     gdv_name = "QSE"

    qasm_file_name = "_private_benchmark/{}/{}QBT_{}CYC_{}_{}.qasm".format(
        folder, nr_qubits, depth_string, name_end, trail)

    # print("qiskit", depth)
    # input qasm file as circuit
    test_circuit = qiskit.QuantumCircuit.from_qasm_file(qasm_file_name)

    """
        NetworkX analysis data

    """
    dag_circ = circuit_to_dag(test_circuit)

    undirect = dag_circ._multi_graph.to_undirected(as_view=True)
    weighted = convert_to_weighted_graph(dag_circ._multi_graph)

    # import matplotlib.pyplot as plt
    # # nx.draw_networkx(undirect, with_labels=False, node_size=10)
    # nx.draw_networkx(dag_circ._multi_graph, with_labels=False, node_size=10)
    # plt.show()

    return max(nx.pagerank(weighted).values()),\
        nx.number_connected_components(undirect),\
        undirect.number_of_edges(),\
            undirect.number_of_nodes(), \
           nx.global_efficiency(weighted), \
           nx.s_metric(dag_circ._multi_graph, False)


def benchmark(depth, trail, parameters):

    if gdv_name == "TFL":
        folder = "BNTF"
        depth_string = "{:02}".format(depth)
        name_end = "TFL"
        if nr_qubits == 54:
            name_end = "QSE"

    elif gdv_name == "QSE":
        folder = "BSS"
        depth_string = "{:03}".format(depth)
        name_end = "QSE"

    # if nr_qubits==54:
    #     gdv_name = "QSE"

    qasm_file_name = "_private_benchmark/{}/{}QBT_{}CYC_{}_{}.qasm".format(
        folder, nr_qubits, depth_string, name_end, trail)

    solution_file_name = "_private_benchmark/meta/{}QBT_{}CYC_{}_{}_solution.csv".format(
        nr_qubits, depth_string, name_end, trail)

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
    # print(original_nodes)

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
    # print("optimal mapping: the circuit has", optimal_depth, "cycles")
    # print(map_original_circuit.draw(style="text"))
    # construct passes that use the DenseLayout+StochasticSwap without initial mapping


    """
       K7M 
    """
    gate_costs = {'id': 0, 'u1': 0, 'measure': 0,
                  'reset': 0, 'barrier': 0,
                  'u2': 1, 'u3': 1, 'U': 1,

                  "ok": 0,
                  # update the costs
                  "rev_cnot": 4 * 1 + 10,  # 4 u2/hadamard + 1 cnot
                  "swap": 3 * 10 + 4,  # 4 u2/hadamard + 3 cnot

                  'seed': 19}  # pass the seed through gate costs


    # parameters_string = str(parameters)

    # the number of qubits in the device
    parameters["nisq_qubits"] = qiskit_coupling_map.size()
    # Add the gate costs
    parameters["gate_costs"] = gate_costs
    # Should the initial mapping be chosen random?
    parameters["initial_map"] = K7MInitialMapping.HEURISTIC
    parameters["unidirectional_coupling"]=False
    parameters["dry_run"] = False

    k7mcomp = K7MCompiler(connection_list[qubits[nr_qubits]], parameters)

    execution_time = time.time()
    map_test_circuit, init_time, init_map = k7mcomp.run(test_circuit)
    execution_time = time.time() - execution_time

    if (map_test_circuit is None) and (init_map is None):
        # this happens when the execution was interrupted
        return optimal_depth, -1, execution_time, init_time, -1, -1

    # print(map_test_circuit.draw(output="text", fold=-1))
    # tmp_circuit = map_test_circuit.decompose()
    tmp_circuit = map_test_circuit
    # print(tmp_circuit.draw(output="text", fold=-1))
    # tmp_circuit = qiskit_to_tk(map_test_circuit)
    # Transform.RebaseToQiskit().DecomposeSWAPtoCX().apply(tmp_circuit)
    depth_result = tmp_circuit.depth()

    # print("k7m mapping: the circuit has", depth_result, "cycles")
    # print(map_test_circuit.draw(style="text"))
    # accumulate result
    # print("----")

    nr_t1 = noOfTranspositions( list(range(nr_qubits)), original_nodes, nr_qubits)
    nr_t2 = noOfTranspositions(original_nodes, init_map, nr_qubits)


    return optimal_depth, depth_result, execution_time, init_time, nr_t1, nr_t2


file_op_type = "a"
if first_run:
    file_op_type = "w"
first_run = False

header = ["max_page_ranke", "nr_conn_comp", "edges", "nodes",
          "efficiency", "smetric",
          "optimal_depth", "res_depth",
          "total_time", "init_time",
          "nr_t1", "nr_t2"]

with open("_private_data/13_training_no_analysis.csv", "w",  buffering=1) as csvFile:
    writer = csv.writer(csvFile)
    # writer.writerow([trail, "k7m", optimal_depth, depth_result, execution_time])


    writer.writerow(header)
    print(*header)

    for m_depth_p in [13]:
        for m_c_p in [1, 3, 5]:
            for b_p in range(0, 21, 2):
                for c_p in range(0, 21, 5):
                    for div_p in range(2, 11, 4):
                        for cx_p in range(2, 11, 4):

                            for trail in range(10):
                                for depth in depth_range[gdv_name]:

                                    parameters = {
                                        "max_depth": m_depth_p,
                                        "max_children": m_c_p,

                                        "att_b": b_p,
                                        "att_c": c_p / 20,

                                        "div_dist": div_p / 10,
                                        "cx": cx_p,

                                        # UNUSED
                                        "opt_att": True,
                                        "opt_max_t_min": False,
                                        "qubit_increase_factor": 3,
                                        "option_skip_cx": False,
                                        "penalty_skip_cx": 20,
                                        "opt_div_by_act": False,
                                        "TIME_LIMIT": 10  # seconds
                                    }

                                    # analysis = circuit_analysis(depth, trail)
                                    analysis = ("n/a", "|")
                                    res =  benchmark(depth, trail, parameters)
                                    params = ("|", m_depth_p, m_c_p, b_p, c_p, div_p, cx_p, depth, trail)
                                    line = analysis + res + params

                                    print(line)
                                    writer.writerow(line)

                                    csvFile.flush()

