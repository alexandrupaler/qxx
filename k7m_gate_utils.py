import sympy
import copy

from qiskit.transpiler.passes import Unroller
from qiskit.qasm.qasm import Qasm

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.dagcircuit import DAGCircuit

# from gatesimplifiers import paler_cx_cancellation, paler_simplify_1q


# TODO clean imports
import qiskit.qasm.node as node

cx_data = {
    "opaque": False,
    "n_args": 0,
    "n_bits": 2,
    "args": [],
    "bits": ["c", "t"],
    # gate cx c,t { CX c,t; }
    "body": node.GateBody([
        node.Cnot([
            node.Id("c", 0, ""),
            node.Id("t", 0, "")
        ])
    ])
}

u2_data = {
    "opaque": False,
    "n_args": 2,
    "n_bits": 1,
    "args": ["phi", "lambda"],
    "bits": ["q"],
    # gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
    "body": node.GateBody([
        node.UniversalUnitary([
            node.ExpressionList([
                node.BinaryOp([
                    node.BinaryOperator('/'),
                    node.Real(sympy.pi),
                    node.Int(2)
                ]),
                node.Id("phi", 0, ""),
                node.Id("lambda", 0, "")
            ]),
            node.Id("q", 0, "")
        ])
    ])
}

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


def add_signatures_to_circuit(dag_circuit):
    dag_circuit.add_basis_element("u2", 1, 0, 2)
    dag_circuit.add_gate_data("u2", u2_data)

    dag_circuit.add_basis_element("h", 1)
    dag_circuit.add_gate_data("h", h_data)

    dag_circuit.add_basis_element("CX", 2)
    dag_circuit.add_basis_element("cx", 2)
    dag_circuit.add_gate_data("cx", cx_data)
    dag_circuit.add_gate_data("CX", cx_data)

    return dag_circuit


def k7m_online_cx_cancellation(circuit, gate):
    '''
    Cancel a CNOT in the circuit if the gate to add is a similar CNOT
    :param circuit: the circuit
    :param gate: the cnot
    :return: True if to add the cnot to the circuit
    '''
    #is this a cnot?
    if gate["name"] not in ["cx", "CX"]:
        return True

    # this a cnot: it has two qargs and no cargs
    # get the predecessors of the output nodes that would touch the two cnot qubits
    out1 = circuit.output_map[gate["qargs"][0]]
    out2 = circuit.output_map[gate["qargs"][1]]
    pred1 = list(circuit.multi_graph.predecessors(out1))
    pred2 = list(circuit.multi_graph.predecessors(out2))

    if len(pred1) != len(pred2):
        return True

    if len(pred1) == len(pred2) and len(pred1) == 1 and pred1[0] == pred2[0]:
        qargs0 = circuit.multi_graph.node[pred1[0]]["qargs"]
        qargs1 = gate["qargs"]
        if qargs0 == qargs1:
            # the gate to add is the same the predecessor
            # delete the predecessor
            circuit._remove_op_node(pred1[0])
            # do not add the current gate
            return False
        # else:
        #     print(qargs0, qargs1)
    # else:
    #     print(pred1, pred2)

    return True


def append_ops_to_dag(dag_circuit, op_list):

    for op in op_list:
        if k7m_online_cx_cancellation(dag_circuit, op):
            # if paler_simplify_1q(dag_circuit, op):
            # Use  Optimize1qGates and CXCancellation from the new qiskit
            dag_circuit.apply_operation_back(op["name"],
                                             op["qargs"],
                                             params=op["params"])

    return dag_circuit.node_counter


def clone_dag_without_gates(dag_circuit):
    compiled_dag = DAGCircuit()
    compiled_dag.basis = copy.deepcopy(dag_circuit.basis)
    compiled_dag.gates = copy.deepcopy(dag_circuit.gates)
    # compiled_dag.add_qreg(QuantumRegister(get_dag_nr_qubits(dag_circuit), "q"))
    # compiled_dag.add_creg(ClassicalRegister(get_dag_nr_qubits(dag_circuit), "c"))

    compiled_dag.add_qreg(QuantumRegister(dag_circuit.num_qubits(), "q"))
    compiled_dag.add_creg(ClassicalRegister(dag_circuit.num_clbits(), "c"))

    return compiled_dag


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


# def get_unrolled_qasm(dag_circuit):
#
#     # print(dag_circuit.qasm())
#
#     basis_gates = "u1,u2,u3,cx,id"  # QE target basis
#     program_node_circuit = Qasm(data=dag_circuit.qasm()).parse()
#     unroller_circuit = Unroller(program_node_circuit,
#                             unroll.DAGBackend(
#                             basis_gates.split(",")))
#
#     return unroller_circuit.execute()