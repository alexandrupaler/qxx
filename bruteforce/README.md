# Evaluation of parameter configurations


## Circuit analysis

The QUEKO TFL 16 qubit circuits are used for these experiments.

`analysis.csv` includes the analysis of the TFL benchmark circuits. For the 
analysis the circuits are treated as graphs.
* `depth` and `trail` are the key for the TFL circuit
* `max_page_rank` is the maximum value of the [page rank](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pagerank)
* `nr_conn_comp` is the number of [connected components](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.number_connected_components.html#networkx.algorithms.components.number_connected_components)
* `edges and nodes` is the number of edges and nodes in the graph
* `efficiency` is the efficiency as [computed in networkx](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.number_connected_components.html#networkx.algorithms.components.number_connected_components)
* `smetric` is another metric [obtained from networkx](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.smetric.s_metric.html#networkx.algorithms.smetric.s_metric)

## CSV file format

`example_training_no_analysis.csv` is an example of the data existing in the
larger CSV files:
* the last two columns, `depth` and `trail`, identify the circuit for which the
heuristic was applied. These are the keys from `analysis.csv`
* `n/a` column is not used
* `|` columns are used only for separation

The following columns from `example_training_no_analysis.csv` are the
results of using K7M for a circuit identified by `depth,trail`:
* `optimal_depth` is know best depth of the circuit
* `res_depth` is the depth obtained by the heuristic
* `total_time` is the time necessary to compute initial mapping and SWAP network
* `init_time` is the time necessary to execute the K7M heuristic for initial mapping. 
The `total_time` includes `init_time`.
* `nr_t1` is the number of transpositions between the identity initial mapping
`1,2,..,q` and the mapping known to generate `optimal_depth`
* `nr_t2` is the number of transpositions between the K7M-computed initial
mapping and the mapping known to generate `optimal_depth`

The following columns from `example_training_no_analysis.csv` are the
parameters used by K7M for a circuit identified by `depth,trail`:
* `max_depth` is the maximum depth of the K7M search tree
* `max_children` is the maximum children per node of the K7M search tree
* `att_b` is the B parameter of the Gaussian from the K7M distance function
* `att_c` is the C parameter of the Gaussian from the K7M distance function
* `div_dist` is the movement factor from the K7M heuristic
* `cx` is the edge cost from the K7M heuristic
