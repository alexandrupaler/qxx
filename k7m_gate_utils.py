# import sympy
import copy
import math

from qiskit.transpiler.passes import Unroller
from qiskit.qasm.qasm import Qasm

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
# from gatesimplifiers import paler_cx_cancellation, paler_simplify_1q

# TODO clean imports
import qiskit.qasm.node as node

def k7m_online_cx_cancellation(circuit, gate):
    '''
    Cancel a CNOT in the circuit if the gate to add is a similar CNOT
    :param circuit: the circuit
    :param gate: the cnot
    :return: True if to add the cnot to the circuit
    '''
    #is this a cnot?
    if gate.name not in ["cx", "CX"]:
        return True

    # this a cnot: it has two qargs and no cargs
    # get the predecessors of the output nodes that would touch the two cnot qubits
    out1 = circuit.output_map[gate.qargs[0]]
    out2 = circuit.output_map[gate.qargs[1]]

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
        # if k7m_online_cx_cancellation(dag_circuit, op):
            # if paler_simplify_1q(dag_circuit, op):
            # Use  Optimize1qGates and CXCancellation from the new qiskit
            # TODO: FIX IT!!!
            # dag_circuit.apply_operation_back(op, op.qargs)
            # dag_circuit.apply_operation_back(op)
            dag_circuit.apply_operation_back(op.op, qargs=op.qargs)


    return dag_circuit.node_counter


def comp_cnot_gate_list(qub1, qub2, quant_reg, inverse_cnot = False):
    ret_params = []

    from qiskit.extensions.standard import HGate, CnotGate
    from qiskit.dagcircuit.dagnode import DAGNode

    if not inverse_cnot:
        ret_params.append([CnotGate(), [quant_reg[qub1], quant_reg[qub2]]])
    else:
        ret_params.append([HGate(), [quant_reg[qub1]]])
        ret_params.append([HGate(), [quant_reg[qub2]]])

        ret_params.append([CnotGate(), [quant_reg[qub1], quant_reg[qub2]]])

        ret_params.append([HGate(), [quant_reg[qub1]]])
        ret_params.append([HGate(), [quant_reg[qub2]]])

    ret = []
    for elem in ret_params:
        ret.append(DAGNode({"op" : elem[0], "qargs" : elem[1], "type" : "op"}))

    return ret


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
                    node.Real(math.pi),
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
                node.Real(math.pi)
            ]),
            node.PrimaryList([
                node.Id("a", 0, "")
            ])
        ])
    ])
}