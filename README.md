## K7M - a method to map circuits to NISQ

K7M is a framework for investigating heuristics to map quantum circuits to NISQ.
In its current implementation is was used to find a good initial mapping of
 qubits such that a low number of SWAPS is introduced.
 
The general idea of K7M is that the initial placement of the qubits influences 
to a great extent the total cost of  compiling a circuit for NISQ. Thus, it 
seems reasonable to:
1) invest more computational power to find an initial placement using a 
lookahead heuristic that estimates as good  as possible the cost of the 
fully mapped circuit;
2) after finalising the initial placement, do not invest too much care and 
computational power to improve the cost of the placement heuristic.


The K7M algorithm were briefly explained and used for some results in
 https://arxiv.org/abs/1811.08985


Please use the following citation, if this software in this repository was useful to your research.

```
@inproceedings{paler2019influence,
  title={On the influence of initial qubit placement during NISQ circuit compilation},
  author={Paler, Alexandru},
  booktitle={International Workshop on Quantum Technology and Optimization Problems},
  pages={207--217},
  year={2019},
  organization={Springer}
}
```

For the moment, K7M can be used as a `TransformationPass` optimizer from Qiskit.
The algorithm is controlled by parameters which are specified in a dictionary.
*TODO: Document the parameters.*

A simple usage example can be seen in [main.py](main.py).


### Original description of K7M


The main objectives of K7M is scalability such that larger circuits can 
also be mapped in reasonble time. The cost of the mapping (after assigning
integer costs for each gate type) should also be reasonable.

A formal description of how an exact mapping algorithm would like 
(based on complete backtracking) was first formulated in order to determine 
the type of the necessary heuristics. The backtracking formulation is
modular, such that the exact algorithm can adapted in a modular manner: 
pre- and post-processing of the circuit, computing the next algorithmic step, 
etc. The modularity of K7M allows to extend/replace available heuristics 
with more advanced ones.

For this particular implementation, local optimisations were preferred to
global ones. The used heuristics are formulated by the motto *minimise a cost
 function given as few details as possible about the search space*
 
 
This variant of K7M tackles the compilation problem as two placement problems: 
1) determine a cost-efficient placement of circuit qubits to architecture qubits;
2) implement using as few SWAPs as possible the CNOTs from the circuit
(place each CNOT on one of the edges of the coupling graph).

For both sub-problems the heuristics use pathfinding methods.

The current heuristics are:
1) The initial placement is chosen such that a global ordering 
of lines/wires/qubits is minimised
2) A Floyd Warshall algorithm is executed on the coupling graph to determine 
all shortest paths between pairs of graph nodes.
3) An unmapped quantum circuit is traversed in topological order, and gates are
mapped to the device in the traversal order.
4) During traversal, each CNOT qubit pair is moved to the coupling graph edge 
that is reached within a minimum distance for the current position of the
circuit qubits.
4) CNOTs are canceled on the fly during circuit compilation
5) Single qubit gates are simplified on the fly during circuit compilation.
The resulting u3 gates are represented as rz.ry.rz matrices and decomposed
into Euler angles.


Some heuristics (for the moment disabled) worked best for grid layouts
(see heuristic_choose_coupling_edge_idx())
a) Clustering: choosing the coupling graph edge closest to previous ones
where CNOTs were executed.
b) Preferring to map circuit CNOTs on the direct edges of the graph
(such that Hadamards are not introduced).

PS: The name of the heuristic is borrowed from the 
[Renault engine](https://en.wikipedia.org/wiki/Renault_K-Type_engine).