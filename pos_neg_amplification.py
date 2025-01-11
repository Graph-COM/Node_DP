import numpy as np
#from poisson_utils import calculate_poisson_rdp_epsilon, calculate_pos_neg_rdp_epsilon, calculate_poisson_rdp_epsilon_integer
import matplotlib.pyplot as plt
#from rdp_opacus import _compute_rdp
from opacus.accountants.rdp import RDPAccountant # original version provided by Opacus
from opacus.accountants.rdp_pos_neg import PosNegRDPAccountant


# setting
n = int(1e6) # num of nodes
E = n * 5 # num of edges
batch_size = 64 # batch size
num_neg_pairs = 4 # num of negative samples per positive sample
K = 5 # maximal node degree
#num_neg_pairs, K = 0, 1 # sanity check, this should give the same result as poisson
gamma = batch_size / E # positive Poisson sampling rate
sigma = 0.43 # Gaussian noise level
num_runs = int( 1 / gamma / 10) # num_of runs to go, only use partial data
delta = 1 / E # privacy budget of ADP delta


poisson_rdp_account = RDPAccountant()
pos_neg_rdp_account = PosNegRDPAccountant(max_node_degree=K, num_neg_pairs=num_neg_pairs, num_nodes=n, num_edges=E)

# execute DP algorithms for num_runs steps
for step in range(num_runs):
    poisson_rdp_account.step(noise_multiplier=sigma, sample_rate=gamma)
    pos_neg_rdp_account.step(noise_multiplier=sigma, sample_rate=gamma)


poisson_eps = poisson_rdp_account.get_epsilon(delta) # optimal adp eps found by searching over a range of alpha
pos_neg_eps = pos_neg_rdp_account.get_epsilon(delta)


print('Noise level sigma=%.3f' % sigma)
print('ADP parameter for poisson subsampling: eps=%.6f, delta=%.2E' % (poisson_eps, delta))
print('ADP parameter for pos-neg subsampling: eps=%.6f, delta=%.2E' % (pos_neg_eps, delta))


