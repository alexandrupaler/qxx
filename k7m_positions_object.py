import numpy
import copy

from k7m_start_configuration import cuthill_order

class PositionsObject:
    def __init__(self, dag_circuit, coupling_object, random=False):

        # if nrq > 6:
        y = cuthill_order(dag_circuit, coupling_object)
        # else:
        # y = list(range(nrq))

        # random = True

        x = numpy.random.permutation(dag_circuit.num_qubits())

        '''
            Current logical qubit position on physical qubits
        '''
        """
            Returns an int-to-int map: logical qubit @ physical qubit map
            There is something similar in Coupling, but I am not using it
        """
        self.pos_circuit_to_phys = {}
        for i in range(dag_circuit.num_qubits()):
            # configuration[i] = nrq - 1 - i # qubit i@i
            self.pos_circuit_to_phys[i] = y[i]  # qubit i@i
            if random:
                self.pos_circuit_to_phys[i] = x[i]

        print("current positions computed", self.pos_circuit_to_phys)

        """
            The reverse dictionary of current_positions
        """
        self.pos_phys_to_circuit = {v: k for k, v in
                                    self.pos_circuit_to_phys.items()}


    def update_configuration(self, route_phys):
        # TODO: FIX IT!!!
        """
        Due to refactoring this a very complicated way to update
        :param route_phys: route of physical qubits on the coupling graph
        :return:
        """
        route_circuit = [self.pos_phys_to_circuit[x] for x in route_phys]

        if len(route_circuit) == 0:
            return

        route_circuit_backup = copy.deepcopy(route_circuit)

        current_mapping = []
        for route_circ_el in route_circuit_backup:
            current_mapping.append(self.pos_circuit_to_phys[route_circ_el])

        """
            Because there is a chain of SWAPs being simulated a chain
            of logical qubits 1,2,3,4,5 will become 2,3,4,5,1
            Thus, route_circuit_backup is a circular permuation
        """
        # Add the first element to the end
        route_circuit_backup.append(route_circuit_backup[0])
        # Simulate the permutation by reading from 1 to end (previous start)
        route_circuit_backup = route_circuit_backup[1:]

        # Update the circuit to phys
        for i in range(0, len(route_circuit_backup)):
            self.pos_circuit_to_phys[route_circuit_backup[i]] = current_mapping[i]

        # Calculate the inverse dictionary - phys to circuit
        self.pos_phys_to_circuit.update({v: k for k, v in self.pos_circuit_to_phys.items()})


    def translate_op_to_coupling_map(self, op):
        '''
            Translate qargs and cargs depending on the position of the logical qubit
        '''

        translated_op = copy.deepcopy(op)

        # translated_op = {}
        # translated_op["name"] = op.name
        # translated_op["type"] = op.type

        # TODO: deepcopy needed?
        # translated_op["params"] = copy.deepcopy(op.params)
        # translated_op["condition"] = copy.deepcopy(op.condition)

        translated_op.qargs = []
        # TODO: FIX IT!!!
        for qubit_args in op.qargs:
            x = self.pos_circuit_to_phys[qubit_args[1]]
            # old code below
            # translated_op.qargs.append(("q", x))
            # new code belo
            translated_op.qargs.append((qubit_args[0], x))

        # TODO: Leave untranslated for the moment?
        # translated_op["cargs"] = copy.deepcopy(op.cargs)

        return translated_op

        # translated_op = {}
        # translated_op["name"] = op.name
        # translated_op["type"] = op.type
        #
        # # TODO: deepcopy needed?
        # translated_op["params"] = copy.deepcopy(op.params)
        # # translated_op["condition"] = copy.deepcopy(op.condition)
        #
        # translated_op["qargs"] = []
        # for qubit_args in op.qargs:
        #     x = self.current_positions[qubit_args[1]]
        #     translated_op["qargs"].append(("q", x))
        #
        # # TODO: Leave untranslated for the moment?
        # translated_op["cargs"] = copy.deepcopy(op.cargs)
        #
        # return translated_op

    def debug_configuration(self):
        print("-------------------")
        for i in range(0, len(self.pos_circuit_to_phys)):
            print("q", i, "@", self.pos_circuit_to_phys[i])


"""
    The following methods were used instead of cuthill in __init__
"""

#     use pagerank
#     import operator
#     pr1 = nx.pagerank(coupling_object["coupling"].G)
#     sorted_pr1 = sorted(pr1.items(), key=operator.itemgetter(1))
# #     print(sorted_pr1)
# #
#     qgraph = qubit_graph(dag_circuit, False)
#     pr2 = nx.pagerank(qgraph)
#     sorted_pr2 = sorted(pr2.items(), key=operator.itemgetter(1))
#     # print(sorted_pr2)
#     for i in range(len(pr1)):
#         configuration[sorted_pr2[i][0]] = sorted_pr1[i][0] - 1

# import operator
# qgraph = qubit_graph(dag_circuit, True)
# g2 = nx.compose(qgraph, coupling_object["coupling"].G)
# pos2 = nx.spectral_layout(g2)
#
# # import matplotlib.pyplot as plt
# # print("pos2", pos2)
# # nx.draw(g2, pos2)
# # plt.show()
#
# coll2 = {}
# for x in pos2:
#     if x <= 0:
#         coll2[-x] = pos2[x][1]
# sorted_3 = sorted(coll2.items(), key=operator.itemgetter(1))
# for i in range(len(sorted_3)):
#     configuration[sorted_3[i][0]] = i

