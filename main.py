import json

from qiskit import QuantumCircuit

from k7m_core import K7MCompiler

basis_gates = 'u1,u2,u3,cx,id'  # or use "U,CX?"

gate_costs = {'id': 0, 'u1': 0, 'measure': 0,
              'reset': 0, 'barrier': 0,
              'u2': 1, 'u3': 1, 'U': 1,
              'cx': 10, 'CX': 10,
              'seed': 19}  # pass the seed through gate costs

def main():

    # Load the circuit
    circ = QuantumCircuit.from_qasm_file("./circuits/random0_n5_d5.qasm")
    print(circ.draw(output="text", fold=-1))

    # Load the coupling map
    # TODO: Test small circuits on large graphs. Was working before refactor.
    name = "./layouts/circle_reg_q16.json"
    with open(name, 'r') as infile:
        temp = json.load(infile)

    coupling = []
    for ii, kk in temp["coupling_map"].items():
        for k in kk:
            coupling += [[int(ii), k]]


    """
        K7M
    """
    parameters = {
        # the number of qubits in the device
        "nisq_qubits" : temp["qubits"],

        # maximum number of children of a node
        "max_children": circ.n_qubits,
        # maximum depth of the search tree
        # after this depth, the leafs are evaluated and only the path with minimum cost is kept in the tree
        # thus, the tree is pruned
        "max_depth": circ.n_qubits,
        # the first number_of_qubits * this factor the search maximises the cost
        # afterwards it minimises it
        "qubit_increase_factor": 3,  # nrq + 1#1.4

        "skipped_cnot_penalty": 200,
    }

    k7m = K7MCompiler(coupling, gate_costs)
    result = k7m.run(circ, parameters)

    print(result.draw(output="text", fold=-1, idle_wires=False))


if __name__ == '__main__':
    main()