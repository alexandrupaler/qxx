import sympy

from qiskit.dagcircuit import DAGCircuit
from qiskit import qasm, unroll
from qiskit.qasm import _node as node

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


def get_unrolled_qasm(dag_circuit):
    basis_gates = "u1,u2,u3,cx,id"  # QE target basis
    program_node_circuit = qasm.Qasm(data=dag_circuit.qasm()).parse()
    unroller_circuit = unroll.Unroller(program_node_circuit,
                            unroll.DAGBackend(
                            basis_gates.split(",")))

    return unroller_circuit.execute()