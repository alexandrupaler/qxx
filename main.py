import json

from qiskit import QuantumCircuit

from k7m_core import K7MCompiler

basis_gates = 'u1,u2,u3,cx,id'  # or use "U,CX?"

def main():

    # Load the circuit
    circ = QuantumCircuit.from_qasm_file("./circuits/random0_n5_d5.qasm")
    print(circ.draw(output="text", fold=-1))

    # Load the coupling map
    # name = "./layouts/circle_reg_q5.json"
    # name = "./layouts/ibmqx3_q16.json"
    name = "./layouts/rect_def_q20.json"
    with open(name, 'r') as infile:
        temp = json.load(infile)

    coupling = []
    for ii, kk in temp["coupling_map"].items():
        for k in kk:
            coupling += [[int(ii), k]]


    """
        K7M
    """
    gate_costs = {'id': 0, 'u1': 0, 'measure': 0,
                  'reset': 0, 'barrier': 0,
                  'u2': 1, 'u3': 1, 'U': 1,

                  'cx': 10, 'CX': 10,
                  "rev_cx_edge" : 10, # related to the edge in the coupling

                  "ok": 0,
                  # update the costs
                  "rev_cnot": 4 * 1 + 10, # 4 u2/hadamard + 1 cnot
                  "swap": 3 * 10 + 4, # 4 u2/hadamard + 3 cnot

                  'seed': 19}  # pass the seed through gate costs

    parameters = {
        # the number of qubits in the device
        "nisq_qubits" : temp["qubits"],

        # maximum number of children of a node
        "max_children": circ.n_qubits,

        # maximum depth of the search tree
        # after this depth, the leafs are evaluated
        # and only the path with minimum cost is kept in the tree
        # thus, the tree is pruned
        "max_depth": circ.n_qubits,

        # the first number_of_qubits * this factor the search maximises the cost
        # afterwards it minimises it
        "option_max_then_min": False,
        "qubit_increase_factor": 3,  # nrq + 1#1.4

        "option_skipped_cnots": False,
        "penalty_skipped_cnot": 200,

        # later changes in the mapping should not affect
        # the initial mapping of the circuit
        "option_attenuate" : True,

        "random_initial" : False,

        "dry_run" : False,
        "gate_costs" : gate_costs,
    }

    k7m = K7MCompiler(coupling, parameters)
    result = k7m.run(circ)

    print(result.draw(output="text", fold=-1, idle_wires=False))


if __name__ == '__main__':
    main()