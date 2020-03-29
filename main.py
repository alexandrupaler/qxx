import json

from k7m_core_functionality import K7MCompiler

basis_gates = 'u1,u2,u3,cx,id'  # or use "U,CX?"

gate_costs = {'id': 0, 'u1': 0, 'measure': 0,
              'reset': 0, 'barrier': 0,
              'u2': 1, 'u3': 1, 'U': 1,
              'cx': 10, 'CX': 10,
              'seed': 19}  # pass the seed through gate costs

def main():

    # Load the circuit
    from qiskit import QuantumCircuit
    circ = QuantumCircuit.from_qasm_file("./circuits/random0_n5_d5.qasm")
    from qiskit.converters import circuit_to_dag

    # Load the coupling map
    # TODO: Test small circuits on large graphs. Was working before refactor.
    name = "./layouts/circle_reg_q5.json"
    with open(name, 'r') as infile:
        temp = json.load(infile)

    coupling = []
    for ii, kk in temp["coupling_map"].items():
        for k in kk:
            coupling += [[int(ii), k]]


    k7m = K7MCompiler(coupling, gate_costs)
    k7m.run(circuit_to_dag(circ))




if __name__ == '__main__':
    main()