# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Generate coupling maps for Complier Tests.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_layout(coupling_layout, scale=1.):
    """
    Plot a coupling layout as a directed graph

    Args:
        coupling_layout (dict):
                dict
                    {
                        "qubits" : the number of qubits in the coupling map
                        "name": coupling map name
                        "coupling_map" : actual coupling map as used by QISKit
                        "position" : arrangement of the qubits in 2D if available (used for plotting)
                        "description": additional information on the coupling map
                    }

        scale (float): scales the graph to help make the plot nicer

    """
    # Use the position information in the coupling_layout to plot the circuit
    import matplotlib.pyplot as plt
    import warnings
    import networkx as nx

    coupling_map = coupling_layout["coupling_map"]
    G = nx.DiGraph()
    ll = list(set([elem for nn in list(coupling_map.values()) for elem in nn] + list(coupling_map.keys())))
    pos = coupling_layout.get("position")
    for qnr in ll:
        if pos == None:
            G.add_node(str(qnr))
        else:
            G.add_node(str(qnr), pos=pos[qnr])
    for qnr in coupling_map:
        for tnr in coupling_map[qnr]:
            G.add_edge(str(qnr), str(tnr), weight=2)
    if pos == None:
        pos = nx.spring_layout(G, k=0.6)
    else:
        pos = nx.get_node_attributes(G, 'pos')
    for dat in pos:
        pos[dat] = np.array(pos[dat]) * scale
    xxx = np.transpose(list(pos.values()))[0]
    yyy = np.transpose(list(pos.values()))[1]
    dx = max(xxx) - min(xxx)
    dx = max([dx * 1.5, 7])
    dy = max(yyy) - min(yyy)
    dy = max([dy * 1.5, 7])
    plt.figure(figsize=(dx, dy))
    with warnings.catch_warnings(): # the version of networkx has a deprecation warning which this hides
        warnings.filterwarnings("ignore", message="The is_string_like function was deprecated")
        nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightgreen', alpha=1.0)
        nx.draw_networkx_edges(G, pos, width=1, arrow=True, edge_color='k')
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def linear(n, order=1.0):
    """
    Creates a linear arrangement of qubits

    Args:
        n (positive integer): number of qubits in coupling map.
        order (float between 0.0  and 1.0): orientation of the coupling.
                        For a value of 0.0 the qubits are coupled from right to left.
                        For a value of 1.0 the qubits are coupled from left to right.
                        For values 0.0 < order < 1.0 random couplings are chosen with
                        probability of left-to-right given by 'order'.

    Returns:
        dict
        {
            "qubits" : the number of qubits in the coupling map
            "name": coupling map name
            "coupling_map" : actual coupling map as used by QISKit
            "position" : arrangement of the qubits in 2D if available (used for plotting)
            "description": additional information on the coupling map
        }
    """
    res = {}
    res["coupling_map"] = {}
    for ii in range(n - 1):
        if np.random.rand() < order:
            if res["coupling_map"].get(ii) == None:
                res["coupling_map"][ii] = [ii + 1]
            else:
                res["coupling_map"][ii] = [ii + 1] + res["coupling_map"].get(ii)
        else:
            if res["coupling_map"].get(ii + 1) == None:
                res["coupling_map"][ii + 1] = [ii]
            else:
                res["coupling_map"][ii + 1] = [ii] + res["coupling_map"].get(ii + 1)
    res["qubits"] = n
    res["name"] = "line_rand_" + str(n)
    res["description"] = "Line of " + str(n) + " qubits with random cx directions and a probability of"+ str(order*100) +"% for coupling from left to right."
    res["position"] = [[ii / n, 0] for ii in range(n)]
    return res


def circle(n, order=1.0):
    """
    Creates a circular arrangement of qubits

    Args:
        n (positive integer): number of qubits in coupling map.
        order (float between 0.0  and 1.0): orientation of the coupling.
                        For a value of 0.0 the qubits are coupled CCW.
                        For a value of 1.0 the qubits are coupled CW.
                        For values 0.0 < order < 1.0 random couplings are chosen with
                        probability of CW coupling given by 'order'.

    Returns:
        dict
        {
            "qubits" : the number of qubits in the coupling map
            "name": coupling map name
            "coupling_map" : actual coupling map as used by QISKit
            "position" : arrangement of the qubits in 2D if available (used for plotting)
            "description": additional information on the coupling map
        }
    """
    res = {}
    res["coupling_map"] = {}
    for ii in range(n):
        if np.random.rand() < order:
            if res["coupling_map"].get(ii) == None:
                res["coupling_map"][ii] = [(ii + 1) % n]
            else:
                res["coupling_map"][ii] = [(ii + 1) % n] + res["coupling_map"].get(ii)
        else:
            if res["coupling_map"].get((ii + 1) % n) == None:
                res["coupling_map"][(ii + 1) % n] = [ii]
            else:
                res["coupling_map"][(ii + 1) % n] = [ii] + res["coupling_map"].get((ii + 1) % n)
    res["qubits"] = n
    res["name"] = "circle_" + str(n)
    res["description"] = "Circle with " + str(n) + " qubits with cx in random direction and a probability of"+ str(order*100) +"% for coupling CW."
    res["position"] = [[np.sin(ii / n * 2 * np.pi), np.cos(ii / n * 2 * np.pi)] for ii in range(n)]
    return res


def rect(n_right, n_down, order=1.0, defects=0):
    """
    Creates a rectangular arrangement of qubits

    Args:
        n_right (positive integer): number of qubits in each row.
        n_down (positive integer): number of qubits in each column.
        order (float between 0.0  and 1.0): orientation of the coupling.
                        For a value of 0.0 the qubits are coupled from right to left and bottom to top.
                        For a value of 1.0 the qubits are coupled from left to right and top to bottom.
                        For values 0.0 < order < 1.0 random couplings are chosen with
                        probability according to the value of 'order'.
        defects (integer): number of defects to introduce in the lattice.
                        A negative number of 'defects' will attempt to remove as many random links (< = abs(defects))
                        as possible without isolating any qubit from the lattice.
                        A positive number of 'defects' will add links to the lattice randmomly until either
                        all-to-all connectivity is reached or the number of added links reaches 'defects'

    Returns:
        dict
        {
            "qubits" : the number of qubits in the coupling map
            "name": coupling map name
            "coupling_map" : actual coupling map as used by QISKit
            "position" : arrangement of the qubits in 2D if available
            "description": additional information on the coupling map
        }
    """
    res = {}
    res["coupling_map"] = {}
    # to the right (fast axis scan)
    for kk in range(n_down):
        for ll in range(n_right - 1):
            ii = kk * n_right + ll
            if np.random.rand() < order:
                if res["coupling_map"].get(ii) == None:
                    res["coupling_map"][ii] = [ii + 1]
                else:
                    res["coupling_map"][ii] = [ii + 1] + res["coupling_map"].get(ii)
            else:
                if res["coupling_map"].get(ii + 1) == None:
                    res["coupling_map"][ii + 1] = [ii]
                else:
                    res["coupling_map"][ii + 1] = [ii] + res["coupling_map"].get(ii + 1)
    # to the right (fast axis scan)
    for kk in range(n_down - 1):
        for ll in range(n_right):
            ii = kk * n_right + ll
            if np.random.rand() < order:
                if res["coupling_map"].get(ii) == None:
                    res["coupling_map"][ii] = [ii + n_right]
                else:
                    res["coupling_map"][ii] = [ii + n_right] + res["coupling_map"].get(ii)
            else:
                if res["coupling_map"].get(ii + n_right) == None:
                    res["coupling_map"][ii + n_right] = [ii]
                else:
                    res["coupling_map"][ii + n_right] = [ii] + res["coupling_map"].get(ii + n_right)
    res["qubits"] = n_right * n_down
    res["name"] = str(n_right) + "x" + str(n_down) + "_lattice_" + str(n_right * n_down)
    res["description"] = "Rectangular lattice with random cx directions (Probability of"+ str(order*100) +"% for reverse coupling)."
    nn = max(n_down, n_right)
    res["position"] = [[ll / nn, -kk / nn] for kk in range(n_down) for ll in range(n_right)]
    return add_defects(res, defects = defects)


def torus(n_right, n_down, order=1.0, defects=0):
    """
    Creates a rectangular arrangement of qubits which is linked back at the edges of the rectangle.
    This can also be represented as a torus.

    Args:
        n_right (positive integer): number of qubits in each row.
        n_down (positive integer): number of qubits in each column.
        order (float between 0.0  and 1.0): orientation of the coupling.
                        For a value of 0.0 the qubits are coupled from right to left and bottom to top.
                        For a value of 1.0 the qubits are coupled from left to right and top to bottom.
                        For values 0.0 < order < 1.0 random couplings are chosen with
                        probability according to the value of 'order'.
        defects (integer): number of defects to introduce in the lattice.
                        A negative number of 'defects' will attempt to remove as many random links (< = abs(defects))
                        as possible without isolating any qubit from the lattice.
                        A positive number of 'defects' will add links to the lattice randmomly until either
                        all-to-all connectivity is reached or the number of added links reaches 'defects'

    Returns:
        dict
        {
            "qubits" : the number of qubits in the coupling map
            "name": coupling map name
            "coupling_map" : actual coupling map as used by QISKit
            "position" : arrangement of the qubits in 2D if available
            "description": additional information on the coupling map
        }
    """
    res = {}
    if n_right > 1 and n_down > 1:
        n = n_right * n_down
        res["coupling_map"] = {}
        # to the right (fast axis scan)
        for kk in range(n_down):
            for ll in range(n_right):
                ii = kk * n_right
                if np.random.rand() < order:
                    if res["coupling_map"].get(ii + ll) == None:
                        res["coupling_map"][ii + ll] = [ii + (ll + 1) % n_right]
                    else:
                        res["coupling_map"][ii + ll] = [ii + (ll + 1) % n_right] + res["coupling_map"].get(ii + ll)
                else:
                    if res["coupling_map"].get(ii + (ll + 1) % n_right) == None:
                        res["coupling_map"][ii + (ll + 1) % n_right] = [ii + ll]
                    else:
                        res["coupling_map"][ii + (ll + 1) % n_right] = [ii + ll] + res["coupling_map"].get(
                            ii + (ll + 1) % n_right)
        # to the right (fast axis scan)
        for kk in range(n_down):
            for ll in range(n_right):
                ii = kk * n_right + ll
                if np.random.rand() < order:
                    if res["coupling_map"].get(ii) == None:
                        res["coupling_map"][ii] = [(ii + n_right) % n]
                    else:
                        res["coupling_map"][ii] = [(ii + n_right) % n] + res["coupling_map"].get(ii)
                else:
                    if res["coupling_map"].get((ii + n_right) % n) == None:
                        res["coupling_map"][(ii + n_right) % n] = [ii]
                    else:
                        res["coupling_map"][(ii + n_right) % n] = [ii] + res["coupling_map"].get((ii + n_right) % n)
        res["qubits"] = n
        res["name"] = str(n_right) + "x" + str(n_down) + "_torus_" + str(n)
        res["description"] = "Torus lattice with random cx directions (Probability of"+ str(order*100) +"% for reverse coupling)."
        res["position"] = [[ll / nn, -kk / nn] for kk in range(n_down) for ll in range(n_right)]
    return add_defects(res, defects = defects)


def ibmqx(index):
    """
    Creates one of the IBM Quantum Experience coupling maps based on the index provided

    Args:
        index (integer between 2 and 5): specify which of the QX chips should be returned

    Returns:
        dict
        {
            "qubits" : the number of qubits in the coupling map
            "name": coupling map name
            "coupling_map" : actual coupling map as used by QISKit
            "position" : arrangement of the qubits in 2D if available
            "description": additional information on the coupling map
        }
    """
    dat = {2: {"name": "ibmqx2", "qubits": 5, "coupling_map": {0: [1, 2], 1: [2], 3: [2, 4], 4: [2]},
               "description": "IBM QX Sparrow: https://ibm.biz/qiskit-ibmqx2",
               "position": [[-1, 1], [1, 1], [0, 0], [1, -1], [-1, -1]]},
           3: {"name": "ibmqx3", "qubits": 16,
               "coupling_map": {0: [1], 1: [2], 2: [3], 3: [14], 4: [3, 5], 6: [7, 11], 7: [10], 8: [7], 9: [8, 10],
                                11: [10], 12: [5, 11, 13], 13: [4, 14], 15: [0, 14]},
               "description": "IBM QX Albatross: https://ibm.biz/qiskit-ibmqx3",
               "position": [[0, 0]] + [[xx, 1] for xx in range(8)] + [[7 - xx, 0] for xx in range(7)]},
           4: {"name": "ibmqx4", "qubits": 5, "coupling_map": {1: [0], 2: [0, 1, 4], 3: [2, 4]},
               "description": "IBM QX Raven: https://ibm.biz/qiskit-ibmqx4",
               "position": [[-1, 1], [1, 1], [0, 0], [1, -1], [-1, -1]]},
           5: {"name": "ibmqx5", "qubits": 16,
               "coupling_map": {1: [0, 2], 2: [3], 3: [4, 14], 5: [4], 6: [5, 7, 11], 7: [10], 8: [7], 9: [8, 10],
                                11: [10], 12: [5, 11, 13], 13: [4, 14], 15: [0, 2, 14]},
               "description": "IBM QX Albatross: https://ibm.biz/qiskit-ibmqx5",
               "position": [[0, 0]] + [[xx, 1] for xx in range(8)] + [[7 - xx, 0] for xx in range(7)]}
           }
    return dat.get(index)

def add_defects(coupling_layout, defects = 0, unidir = False):
    if defects > 0:
        coupling_layout = __add_links(coupling_layout, round(defects), unidir = unidir)
    if defects < 0:
        coupling_layout = __sub_links(coupling_layout, round(abs(defects)))
    return coupling_layout


def save_coupling(coupling_layout):
    """
    Saves the coupling map that is given as a json file to a subfolder named layouts

    Args:
        dict
        {
            "qubits" : the number of qubits in the coupling map
            "name": coupling map name
            "coupling_map" : actual coupling map as used by QISKit
            "position" : arrangement of the qubits in 2D if available
            "description": additional information on the coupling map
        }

    """
    import json
    from os import path, getcwd
    with open(path.join(getcwd(), "layouts", coupling_layout["name"] + ".json"), 'w') as outfile:
        json.dump(coupling_layout, outfile)

def load_coupling(name):
    """
    Saves the coupling map that is given as a json file to a subfolder named layouts

    Args:
        name (string): name of the coupling map that was used when saving
                        (corresponds to the filename without the extension .json)
    Returns:
        dict
        {
            "qubits" : the number of qubits in the coupling map
            "name": coupling map name
            "coupling_map" : actual coupling map as used by QISKit
            "position" : arrangement of the qubits in 2D if available
            "description": additional information on the coupling map
        }

    """
    import json
    from os import path, getcwd
    with open(path.join(getcwd(),"layouts",name+".json"), 'r') as infile:
        temp = json.load(infile)
        temp["coupling_map"] = {int(ii): kk for ii, kk in temp["coupling_map"].items()}
        return temp


def __add_links(coupling_layout, nl = 0, unidir = False):
    # Add nl links to an existing coupling map
    # If it already exists then choose the next non existing link until full connectivity is reached.
    # TODO check for 'deepcopy' errors. I'm probably directly modifying the coupling_layout and may not have to return anything
    # TODO check if added links already exist in other direction and disallow if unidir = True.
    coupling_map = coupling_layout["coupling_map"]
    n = coupling_layout["qubits"]
    for kk in range(nl):
        # create a random link
        ii = round(np.random.rand()*n-0.5)%n # choose a random source
        jj = round(np.random.rand()*n-0.5)%n # choose a random target
        # if it is a link onto itself shift the target by 1
        if ii == jj:
            jj = (jj + 1) % n
        # check if the source node exists in the coupling map
        if ii in coupling_map:
            # source node is in coupling map
            # store initial node indices
            ii_orig = (ii-1) % n
            jj_orig = jj
            # search the target nodes until we find one that is not linked
            while jj in coupling_map[ii]:
                # the node is already in the list so increase the node number by 1
                jj = (jj + 1) % n
                if jj == jj_orig:
                    # if the target node is again the original node number, increase the source node
                    ii = (ii + 1) % n
                    if ii not in coupling_map:
                        # if the source node is new add  the link
                        # but check first if it points to itself
                        if ii == jj:
                            jj = (jj + 1) % n
                        coupling_map[ii] = [jj]
                        jj = -jj # prepare to exit from the while loop
                    elif ii == ii_orig or n*(n-1) == sum([len(jj) for jj in coupling_map.values()]):
                        # if the increase in the source qubit index has brought us back to where we started
                        # we can assume that all-to-all connectivity was reached
                        coupling_layout["coupling_map"] = coupling_map
                        #coupling_layout["description"] = coupling_layout["description"] + " All-to-all connectivity."
                        return coupling_layout # no more links left to make
                    elif ii == jj:
                            jj = (jj + 1) % n
                            jj_orig = jj
            if jj >= 0:
                coupling_map[ii] = coupling_map[ii] + [jj]
        else:
            # source node is not yet in coupling map so just add the node
            coupling_map[ii] = [jj]
    coupling_layout["coupling_map"] = coupling_map
    #coupling_layout["description"] = coupling_layout["description"]+" Added "+str(nl)+" links to lattice."
    return coupling_layout

def __sub_links(coupling_layout, nl = 0):
    # Remove nl links from an existing coupling map until removal of more links
    # would lead to disjoint sets of qubits
    # TODO check that this works for all cases
    from copy import deepcopy
    coupling_map = coupling_layout["coupling_map"]
    n = coupling_layout["qubits"]
    for kk in range(nl):
        retry = n
        while retry > 0:
            retry = retry - 1
            cmap_copy = deepcopy(coupling_map)
            if len(cmap_copy) > 0:
                # TODO check interval boundaries for rand() function to avoid mod
                ii = list(cmap_copy.keys())[round(np.random.rand() * len(cmap_copy)-0.5) % len(cmap_copy)]
                if len(cmap_copy[ii]) > 1:
                    jj = round(np.random.rand() * len(cmap_copy[ii])-0.5) % len(cmap_copy[ii])
                    del(cmap_copy[ii][jj])
                else:
                    del(cmap_copy[ii])
            if not __is_disjoint(cmap_copy,n):
                coupling_map = cmap_copy
                retry = -10
        if retry == 0:
            coupling_layout["coupling_map"] = coupling_map
            #coupling_layout["description"] = coupling_layout["description"]+" Removed "+str(kk)+" links from lattice."
            return coupling_layout
    coupling_layout["coupling_map"] = coupling_map
    #coupling_layout["description"] = coupling_layout["description"]+" Removed "+str(nl)+" links from lattice."
    return coupling_layout

def __is_disjoint(coupling_map ,nmax):
    # check if all nodes in the map are connected
    # TODO this should be improved and checked if it works in all cases
    # first check if all nodes are present in the coupling map
    if len(set([ii for jj in [list(coupling_map.keys())]+list(coupling_map.values()) for ii in jj])) < nmax:
        return True
    else:
        f_map = np.zeros(nmax) # empty filling map
        f_val = 1
        f_ind = 0 # start flooding from node 0
        f_map[f_ind] = f_val
        while min(f_map) == 0 and f_val < nmax:
            # determine all links from node f_ind not taking directionality into account
            # it includes the backwards link but not the node itself
            f_links = [jj if ii == f_ind else [ii] for (ii,jj) in coupling_map.items() if ii == f_ind or f_ind in jj]
            f_links = [jj for ii in f_links for jj in ii]
            # choose smallest filling value node
            f_min = nmax # initialize with a large value
            for ii in f_links: # find linked node with smallest flooding level if multiple choose first one
                f_val = f_map[ii]
                if  f_val < f_min:
                    f_min = f_val
                    f_ind = ii
            f_map[f_ind] = f_map[f_ind]+1 # increase flooding level at index f_ind
        return (min(f_map) == 0) # return true if there are unflooded nodes in flooding map

