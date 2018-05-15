# Import your solution function
from challenge_submission import compiler_function

# Import submission evaluation and scoring functions
from challenge_evaluation import evaluate, score, qasm_to_dag_circuit

import os

# Possibly useful other helper function
from challenge_evaluation import qasm_to_dag_circuit, load_coupling, get_layout

# Select the simulation backend to calculate the quantum states resulting from the circuits
# On Windows platform the C++ Simulator is not yet available with pip install
backend = 'local_qiskit_simulator'

generateJSONFile = False
computeScore = True

gate_costs2 = {'id': 0, 'u1': 0, 'measure': 0, 'reset': 0, 'barrier': 0,
                  'u2': 1, 'u3': 1, 'U': 1,
                  'cx': 10, 'CX': 10}

if computeScore:
    if generateJSONFile:
        myres = score(compiler_function, backend=backend)
        print("Your compiler scored %6.5f x better \
        and was %6.5f x faster than the QISKit reference compiler." % myres)

        os.remove("run_once_results.json")

    myres = score(compiler_function, backend = backend)
    print("Your compiler scored %6.5f x better \
    and was %6.5f x faster than the QISKit reference compiler." % myres)
else:
    qasm = ""
    with open("./circuits/random0_n16_d16.qasm", "r") as f:
        qasm = f.read()

    cm = load_coupling("ibmqx5_q16")
    compiler_function(qasm_to_dag_circuit(qasm), cm["coupling_map"], gate_costs2)
