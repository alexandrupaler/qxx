import numpy as np
import networkx as nx
import sympy

from sympy import Number as N
from qiskit.mapper._compiling import rz_array, ry_array, euler_angles_1q

def paler_cx_cancellation(circuit, gate):
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


def paler_simplify_1q(circuit, gate):
    '''
    Multiply single qubit gates together
    :param circuit:
    :param gate: gate analysed. It can be a CNOT, then not considered
    :return: if the gate should be added to the circuit
    '''
    #is this a single qubit gate?
    if gate["name"] not in ["u1", "u2", "u3"]:
        return True

    # this a single qubit gate: it has a single qargs and no cargs
    # get the predecessor of the output nodes that would touch the gate
    out1 = circuit.output_map[gate["qargs"][0]]
    pred1 = list(circuit.multi_graph.predecessors(out1))

    # is this a single qubit gate?
    if circuit.multi_graph.node[pred1[0]]["name"] not in ["u1", "u2", "u3"]:
        return True

    modified_from_qiskit(circuit, [pred1[0]], gate)

    return False

def modified_from_qiskit(unrolled, run, gate):
    node_right = unrolled.multi_graph.node[run[0]]

    right_name = node_right["name"]
    right_parameters = (N(0), N(0), N(0))  # (theta, phi, lambda)
    if right_name == "u1":
        right_parameters = (N(0), N(0), node_right["params"][0])
    elif right_name == "u2":
        right_parameters = (sympy.pi / 2, node_right["params"][0], node_right["params"][1])
    elif right_name == "u3":
        right_parameters = tuple(node_right["params"])

    left_name = gate["name"]
    assert left_name in ["u1", "u2", "u3", "id"], "internal error"
    if left_name == "u1":
        left_parameters = (N(0), N(0), gate["params"][0])
    elif left_name == "u2":
        left_parameters = (sympy.pi / 2, gate["params"][0], gate["params"][1])
    elif left_name == "u3":
        left_parameters = tuple(gate["params"])
    else:
        left_name = "u1"  # replace id with u1
        left_parameters = (N(0), N(0), N(0))

    # Compose gates
    name_tuple = (left_name, right_name)
    if name_tuple == ("u1", "u1"):
        # u1(lambda1) * u1(lambda2) = u1(lambda1 + lambda2)
        right_parameters = (N(0), N(0), right_parameters[2] +
                            left_parameters[2])
    # elif name_tuple == ("u1", "u2"):
    #     # u1(lambda1) * u2(phi2, lambda2) = u2(phi2 + lambda1, lambda2)
    #     right_parameters = (sympy.pi / 2, right_parameters[1] +
    #                         left_parameters[2], right_parameters[2])
    # elif name_tuple == ("u2", "u1"):
    #     # u2(phi1, lambda1) * u1(lambda2) = u2(phi1, lambda1 + lambda2)
    #     right_name = "u2"
    #     right_parameters = (sympy.pi / 2, left_parameters[1],
    #                         right_parameters[2] + left_parameters[2])
    # elif name_tuple == ("u1", "u3"):
    #     # u1(lambda1) * u3(theta2, phi2, lambda2) =
    #     #     u3(theta2, phi2 + lambda1, lambda2)
    #     right_parameters = (right_parameters[0], right_parameters[1] +
    #                         left_parameters[2], right_parameters[2])
    # elif name_tuple == ("u3", "u1"):
    #     # u3(theta1, phi1, lambda1) * u1(lambda2) =
    #     #     u3(theta1, phi1, lambda1 + lambda2)
    #     right_name = "u3"
    #     right_parameters = (left_parameters[0], left_parameters[1],
    #                         right_parameters[2] + left_parameters[2])
    # elif name_tuple == ("u2", "u2"):
    #     # Using Ry(pi/2).Rz(2*lambda).Ry(pi/2) =
    #     #    Rz(pi/2).Ry(pi-2*lambda).Rz(pi/2),
    #     # u2(phi1, lambda1) * u2(phi2, lambda2) =
    #     #    u3(pi - lambda1 - phi2, phi1 + pi/2, lambda2 + pi/2)
    #     right_name = "u3"
    #     right_parameters = (sympy.pi - left_parameters[2] -
    #                         right_parameters[1], left_parameters[1] +
    #                         sympy.pi / 2, right_parameters[2] +
    #                         sympy.pi / 2)
    # elif name_tuple[1] == "nop":
    #     right_name = left_name
    #     right_parameters = left_parameters
    else:
        # For composing u3's or u2's with u3's, use
        # u2(phi, lambda) = u3(pi/2, phi, lambda)
        # together with the qiskit.mapper.compose_u3 method.
        right_name = "u3"
        # Evaluate the symbolic expressions for efficiency
        left_parameters = tuple(map(lambda x: x.evalf(), list(left_parameters)))
        right_parameters = tuple(map(lambda x: x.evalf(), list(right_parameters)))
        right_parameters = paler_compose_u3(left_parameters[0],
                                            left_parameters[1],
                                            left_parameters[2],
                                            right_parameters[0],
                                            right_parameters[1],
                                            right_parameters[2])
            # Why evalf()? This program:
            #   OPENQASM 2.0;
            #   include "qelib1.inc";
            #   qreg q[2];
            #   creg c[2];
            #   u3(0.518016983430947*pi,1.37051598592907*pi,1.36816383603222*pi) q[0];
            #   u3(1.69867232277986*pi,0.371448347747471*pi,0.461117217930936*pi) q[0];
            #   u3(0.294319836336836*pi,0.450325871124225*pi,1.46804720442555*pi) q[0];
            #   measure q -> c;
            # took >630 seconds (did not complete) to optimize without
            # calling evalf() at all, 19 seconds to optimize calling
            # evalf() AFTER compose_u3, and 1 second to optimize
            # calling evalf() BEFORE compose_u3.
        # 1. Here down, when we simplify, we add f(theta) to lambda to
        # correct the global phase when f(theta) is 2*pi. This isn't
        # necessary but the other steps preserve the global phase, so
        # we continue in that manner.
        # 2. The final step will remove Z rotations by 2*pi.
        # 3. Note that is_zero is true only if the expression is exactly
        # zero. If the input expressions have already been evaluated
        # then these final simplifications will not occur.
        # TODO After we refactor, we should have separate passes for
        # exact and approximate rewriting.

        '''
            For the competition, maybe not needed
            u1 has cost zero
            u2 and u3 have the same cost
        '''
        # # Y rotation is 0 mod 2*pi, so the gate is a u1
        # if (right_parameters[0] % (2 * sympy.pi)).is_zero \
        #         and right_name != "u1":
        #     right_name = "u1"
        #     right_parameters = (0, 0, right_parameters[1] +
        #                         right_parameters[2] +
        #                         right_parameters[0])
        # # Y rotation is pi/2 or -pi/2 mod 2*pi, so the gate is a u2
        # if right_name == "u3":
        #     # theta = pi/2 + 2*k*pi
        #     if ((right_parameters[0] - sympy.pi / 2) % (2 * sympy.pi)).is_zero:
        #         right_name = "u2"
        #         right_parameters = (sympy.pi / 2, right_parameters[1],
        #                             right_parameters[2] +
        #                             (right_parameters[0] - sympy.pi / 2))
        #     # theta = -pi/2 + 2*k*pi
        #     if ((right_parameters[0] + sympy.pi / 2) % (2 * sympy.pi)).is_zero:
        #         right_name = "u2"
        #         right_parameters = (sympy.pi / 2, right_parameters[1] +
        #                             sympy.pi, right_parameters[2] -
        #                             sympy.pi + (right_parameters[0] +
        #                                         sympy.pi / 2))
        # # u1 and lambda is 0 mod 2*pi so gate is nop (up to a global phase)
        # if right_name == "u1" and (right_parameters[2] % (2 * sympy.pi)).is_zero:
        #     right_name = "nop"

        # Simplify the symbolic parameters
        right_parameters = tuple(map(sympy.simplify, list(right_parameters)))

    # Replace the data of the first node in the run
    new_params = []
    if right_name == "u1":
        new_params = [right_parameters[2]]
    if right_name == "u2":
        new_params = [right_parameters[1], right_parameters[2]]
    if right_name == "u3":
        new_params = list(right_parameters)

    nx.set_node_attributes(unrolled.multi_graph, name='name',
                           values={run[0]: right_name})
    # params is a list of sympy symbols
    nx.set_node_attributes(unrolled.multi_graph, name='params',
                           values={run[0]: new_params})

    # # Delete the other nodes in the run
    # for current_node in run[1:]:
    #     unrolled._remove_op_node(current_node)
    if right_name == "nop":
        unrolled._remove_op_node(run[0])

def paler_yzy_to_zyz2(xi, theta1, theta2, eps=1e-9):
    arr = np.dot(ry_array(theta1), np.dot(rz_array(xi), ry_array(theta2)))

    y1, z1, z2, x = euler_angles_1q(arr)

    return y1, z1, z2


def paler_compose_u3(theta1, phi1, lambda1, theta2, phi2, lambda2):
    """Return a triple theta, phi, lambda for the product.

    u3(theta, phi, lambda)
       = u3(theta1, phi1, lambda1).u3(theta2, phi2, lambda2)
       = Rz(phi1).Ry(theta1).Rz(lambda1+phi2).Ry(theta2).Rz(lambda2)
       = Rz(phi1).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda2)
       = u3(theta', phi1 + phi', lambda2 + lambda')

    Return theta, phi, lambda.
    """
    # Careful with the factor of two in yzy_to_zyz
    # thetap, phip, lambdap = copied_yzy_to_zyz2((lambda1 + phi2) / 2,
    #                                    theta1 / 2, theta2 / 2)
    # (theta, phi, lamb) = (2 * thetap, phi1 + 2 * phip, lambda2 + 2 * lambdap)
    # return (theta.simplify(), phi.simplify(), lamb.simplify())

    # from sympy to normal floats
    a = float((lambda1 + phi2).evalf())
    b = float(theta1.evalf())
    c = float(theta2.evalf())

    thetap, phip, lambdap = paler_yzy_to_zyz2(a, b, c)

    (theta, phi, lamb) = (thetap, phi1 + phip, lambda2 + lambdap)

    # go back to sympy
    return N(theta), N(phi), N(lamb)
