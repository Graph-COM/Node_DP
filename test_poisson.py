import numpy as np
from poisson_utils import calculate_poisson_rdp_epsilon, calculate_pos_neg_rdp_epsilon
import matplotlib.pyplot as plt


# setting
n = int(1e6) # num of nodes
E = n * 5 # num of edges
batch_size = 64 # batch size
num_neg_pairs = 4 # num of negative samples per positive sample
K = 5 # maximal node degree
gamma = batch_size / E # positive Poisson sampling rate
sigma = 0.45 # Gaussian noise level
num_runs = int(1 / gamma) / 2 # num_of runs to go, only use partial data
delta = 1 / E # privacy budget of ADP delta

eps_pos_neg = [] # RDP eps of positive+negative sampling
eps_poisson = [] # RDP eps of poisson subsampling
eps = [] # RDP eps of standard Gaussian mechanism
alphas = np.linspace(2, 50, 200) # range of alpha
for alpha in alphas:
    eps.append(alpha/2/sigma**2)
    eps_poisson.append(calculate_poisson_rdp_epsilon(alpha, gamma, sigma)/(alpha-1)) # note that we need to manually divide by alpha-1
    eps_pos_neg.append(calculate_pos_neg_rdp_epsilon(alpha, gamma, sigma, K, num_neg_pairs, n, E))


eps = np.array(eps)
eps_poisson = np.array(eps_poisson)
eps_pos_neg = np.array(eps_pos_neg)

fig, ax = plt.subplots()
# plog ADP eps profile w.r.t alpha. Remember ADP eps = RDP eps + log(1/delta)/(alpha-1)
ax.plot(alphas, eps+np.log(1/delta)/(alphas-1), label='eps')
ax.plot(alphas, eps_poisson+np.log(1/delta)/(alphas-1), label='eps_poisson')
ax.plot(alphas, eps_poisson*num_runs+np.log(1/delta)/(alphas-1), label='eps_poisson_compose', linestyle='--')
ax.plot(alphas, eps_pos_neg+np.log(1/delta)/(alphas-1), label='eps_pos_neg')
ax.plot(alphas, eps_pos_neg*num_runs+np.log(1/delta)/(alphas-1), label='eps_pos_neg_compose', linestyle='--')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('ADP-eps')
plt.title('ADP-eps(alpha)')
plt.legend()
plt.show()



