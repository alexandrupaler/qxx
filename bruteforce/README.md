# CSV files for benchmarking with different parameter configurations

The QUEKO TFL 16 qubit circuits are used for these experiments.

`analysis.csv` includes the analysis of the TFL benchmark circuits. For the 
analysis the circuits are treated as graphs.
* depth and trail are the key for the TFL circuit
* max_page_rank is the maximum value of the [page rank](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pagerank)
* nr_conn_comp is the number of [connected components](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.number_connected_components.html#networkx.algorithms.components.number_connected_components)
* edges and nodes is the number of edges and nodes in the graph
* efficiency is the efficiency as [computed in networkx](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.number_connected_components.html#networkx.algorithms.components.number_connected_components)
* smetric another metric [obtained from networkx](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.smetric.s_metric.html#networkx.algorithms.smetric.s_metric)

