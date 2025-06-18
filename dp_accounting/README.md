# Privacy Accountant for Two-Stage (Positive/Negative) Sampling

This repository provides a privacy accounting utility tailored to positive-negative sampling strategies, with a focus on sensitivity amplification in graph-based relational learning. The core algorithm implements Rényi Differential Privacy (RDP) accounting under both standard and adaptive gradient clipping assumptions.

## Repository Contents
- pos_neg_amplification.py: Main entry point for running the RDP accountant with positive-negative sampling.
- poisson_utils.py: Helper functions for working with Poisson subsampling.
- rdp_opacus.py: Wrapper and extensions to Opacus’ RDP accountant utilities.

## Usage
To compute the privacy cost under positive-negative sampling, run:
`$ python pos_neg_amplification.py`

The script includes comments and parameter descriptions to guide customization.

The main function of interest is:  
`PosNegRDPAccountant(
    max_node_degree=K,
    num_neg_pairs=num_neg_pairs,
    num_nodes=n,
    num_edges=E,
    constant_sensitivity=False
)`

**Parameters**
- max_node_degree: Maximum node degree (K) used in preprocessing the input graph.
- num_neg_pairs: Number of negative samples (K_neg) per positive sample.
- num_nodes: Total number of nodes in the dataset.
- num_edges: Total number of edges in the dataset.
- constant_sensitivity:
	- False (default): Uses standard per-term gradient clipping, resulting in linearly growing sensitivity with respect to the number of removals.
	- True: Applies adaptive clipping, assuming constant sensitivity across the training process.

**Notes**
When constant_sensitivity=False, the accountant performs tight bound calculations for sensitivity growth, which involves computing univariate Gaussian integrals and can be computationally heavy.
