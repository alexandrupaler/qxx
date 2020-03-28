## K7M - an algorithm to map circuits to NISQ

K7M is a heuristic to find a good initial mapping of qubits such that an small 
number of SWAPS is introduced.

The initial placement of the qubits influences the total cost of compiling a 
circuit for NISQ. The heuristic and lookahead algorithm were used for 
https://arxiv.org/abs/1811.08985

The algorithm was very fast for large circuits. 

The code is uncommented and based on a very old version of Qiskit.
Transitioning to newer Qiskit is currently WIP.