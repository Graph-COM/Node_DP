### Usage
To do privacy accountant, please run `pos_neg_amplification.py`, with comments and parameters explained in it. 

Particularly, the function `PosNegRDPAccountant(max_node_degree=K, num_neg_pairs=num_neg_pairs, num_nodes=n, num_edges=E,constant_sensitivity=False)` realize the accountant for positive-negative sampling. The parameter `constant_sensitivity=False` means we adopt standard per-term gradient clipping and thus the sensitivity grows linearly with the number of removals. In constrast, `constant_sensitivity=True`  means we adopt adaptive clipping and the sensitivity is always a constants, the default setting of this function.

Currently, the tight bound calculation for linear sensitivity requires a heavy computation of univarite Gaussian integral. Though the integral region is heuristically restricted to speed up, the overall runtime is still kind of long.
