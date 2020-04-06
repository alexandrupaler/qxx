"""
This class represents, for the moment, something more than the CouplingMap
from Qiskit. Initially it was dictionary with different fields, in a design
that resembled prehistoric Qiskit.
"""

import networkx as nx
import copy
import collections
import math

from qiskit.transpiler import CouplingMap

class K7MCoupling:
    def __init__(self, coupling_map, parameters):

        self.coupling_map = coupling_map
        self.coupling = CouplingMap(coupling_map)

        """
            Reverse edges are added, assuming that 
            these do not exist in the coupling
            Theoretically, reverse edges could have a different cost.
        """
        self.add_reverse_edges_and_weights(parameters["gate_costs"])

        '''
            Prepare the Floyd Warshall graph and weight matrix
            The graph is directed
        '''
        self.coupling_pred, self.coupling_dist = \
            nx.floyd_warshall_predecessor_and_distance(
                self.coupling.graph,
                weight="weight")


        self.coupling_edges_list = [
            e for e in self.coupling.graph.edges()
        ]

    def heuristic_choose_coupling_edge(self, qub1_to_index, qub2_to_index, next_nodes=[]):
        """
            Heuristic: which coupling edge generates the smallest
            cost given qub1 and qub2 positions
            Returns: the total cost of moving qub1 and qub2 and interacting them
        """

        ret_idx = -1

        min_cost = math.inf
        idx = -1

        for edge in self.coupling_edges_list:
            idx += 1

            edge1 = edge[0]
            edge2 = edge[1]

            cost1 = self.coupling_dist[qub1_to_index][edge1]
            cost2 = self.coupling_dist[qub2_to_index][edge2]

            # do not consider interaction cost?
            tmp_cost = cost1 + cost2  # + self.coupling_dist[qub1_to_index][qub2_to_index]

            '''
                A kind of clustering heuristic:
                the closer the edge is to previous CNOTs, the better (?)
            '''
            # f_idx = len(next_nodes)
            # for node in next_nodes:
            #     tmp_cost += 0.05 * f_idx * self.coupling_dist[node][edge1]
            #     tmp_cost += 0.05 * f_idx * self.coupling_dist[node][edge2]
            #     f_idx -= 1

            '''
                A kind of preference heuristic: 
                prefer edges with the direction of the cnot to execute
            '''
            # TODO: disabled in the add_reverse_edges
            # The type of edge is indicated by its weight. In this case 14
            # is for CNOTs in the reverse direction, because
            # add_reverse_edges_and_weights added 14 (10 + 4 Hadamard)
            # if the CNOT was reversed
            # if self.coupling_dist[edge1][edge2] == 14:
            #     tmp_cost *= 1.1

            if tmp_cost <= min_cost:
                min_cost = tmp_cost
                ret_idx = idx

        # print(min_cost)

        # Determine the indices of the edge nodes where the qubits will be moved
        stop_node_idx1 = self.coupling_edges_list[ret_idx][0]
        stop_node_idx2 = self.coupling_edges_list[ret_idx][1]

        # return ret_idx
        return stop_node_idx1, stop_node_idx2


    def is_pair(self, qubit1, qubit2):

        # pair_pass = (qubit1 in coupling_map)
        # if pair_pass:
        #     pair_pass &= qubit2 in coupling_map[qubit1]
        # return pair_pass

        # bool_found = (qubit1 in self.coupling_map)
        # if bool_found:
        #     bool_found = bool_found and (
        #                 qubit2 in self.coupling_map[qubit1])

        bool_found = [qubit1, qubit2] in self.coupling_map
        bool_found = bool_found or (qubit1, qubit2) in self.coupling_map
        return bool_found


    def add_reverse_edges_and_weights_one(self):
        # # get all edges from coupling
        # edgs = copy.deepcopy(self.coupling.graph.edges)
        # for edg in edgs:
        #     self.coupling.graph.remove_edge(*edg)
        #     # 31.08.2018
        #     self.coupling.graph.add_edge(edg[0], edg[1], weight=1)
        #     self.coupling.graph.add_edge(edg[1], edg[0], weight=1)
        self.add_reverse_edges_and_weights(gatecosts = {"cx": 1})


    def add_reverse_edges_and_weights(self, gatecosts):
        # get all edges from coupling
        edgs = copy.deepcopy(self.coupling.graph.edges)
        for edg in edgs:
            # print(edg)
            self.coupling.graph.remove_edge(*edg)

            # the direct edge gets a weight
            self.coupling.graph.add_edge(edg[0], edg[1], weight = gatecosts["cx"])

            # the inverse edge
            # CNOT + four Hadamards for the reverse
            # coupling.graph.add_edge(edg[1], edg[0],
            # weight=gatecosts["cx"] + 4*gatecosts["u2"])
            self.coupling.graph.add_edge(edg[1], edg[0], weight = gatecosts["rev_cx_edge"])


    def reconstruct_route(self, start_phys, stop_phys):
        """
        Given two vertices start and stop, compute the path between them
        :param start_phys:
        :param stop_phys:
        :return: the list of vertices for the start->stop path
        """
        route = collections.deque()

        if self.coupling_pred[start_phys][stop_phys] is None:
            # can/should this happen?
            return []
        else:
            # route.append(stop)
            route.appendleft(stop_phys)
            while start_phys != stop_phys:
                stop_phys = self.coupling_pred[start_phys][stop_phys]
                # route.append(stop)
                route.appendleft(stop_phys)

        # return list(reversed(route))
        return list(route)


