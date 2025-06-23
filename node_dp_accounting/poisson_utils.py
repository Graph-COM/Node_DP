import numpy as np
from scipy.stats import binom
from scipy.special import erfc
from scipy.optimize import minimize, brentq
from rdp_opacus import _compute_rdp


def log_sum_exp(x):
    # x is an array, return log(exp(x_1)+...+exp(x_n))
    xmax = x.max()
    if xmax == -np.inf: # in case all x are -inf
        return -np.inf
    else:
        return np.log(sum(np.exp(x-xmax))) + xmax # a numerically stable way

def log_generalized_binom_coeff(n, m):
    # the generalized binomial coefficient n*(n-1)*...(n-m+1)/m!
    # taking the log it becomes log(n)+log(n-1)+...+log(n-m+1)-log(m)-log(m-1)-...-log(1)
    if m == 0 or m == n:
        return 0
    elif m > n:
        return -np.inf
    else:
        temp1 = np.log(np.array([n-i for i in range(m)]))
        temp2 = np.log(np.array([i+1 for i in range(m)]))
        res = sum(temp1) - sum(temp2)
        return res

def generalized_binom_coeff(n, m):
    return np.exp(log_generalized_binom_coeff(n, m))


#def generalized_binom_pmf(m, n, gamma):
    #return np.exp(log_generalized_binom_coeff(n, m)+(n-m)*np.log(1-gamma)+m*np.log(gamma))


def gaussian_rdp_epsilon(sigma):
    # the RDP epsilon of Gaussian mechanism as a function of alpha: eps(alpha)= alpha/2/sigma**2
    return lambda alpha: alpha / (2*sigma**2)


def calculate_poisson_rdp_epsilon_integer(alpha, gamma, sigma):
    # compute the RDP epsilon of Poisson subsampling, with sampling rate gamma, gaussian noise sigma and an INTEGER alpha

    # eps = log(C0 + sum_l w_l * exp(e(l)))
    # exp(e(l)) could be too large to store, so we first compute: log(C_0), log(w_l)+e(l), and do log-sum-exp of them
    gaussian_epsilon_function = gaussian_rdp_epsilon(sigma)
    log_C0 = (alpha-1)*np.log(1-gamma)+np.log(alpha*gamma-gamma+1)
    weight = binom.pmf(np.arange(0, alpha+1), alpha, gamma) # compute w_l
    log_weight = np.log(weight)
    exp_factor = np.array([(l - 1) * gaussian_epsilon_function(l) for l in range(2, alpha + 1)]) # compute e(l)
    exp_factor = exp_factor + log_weight[2:]
    res = log_sum_exp(np.concatenate((np.array([log_C0]), exp_factor)))
    return res


def calculate_poisson_rdp_epsilon(alpha, gamma, sigma, conv_delta=1e-10):
    # compute the RDP epsilon of Poisson subsampling, with sampling rate gamma, gaussian noise sigma and any real alpha
    # conv_delta is the convergence criterion for the infinite sums

    # eps consists of two infinite sum, which we will do separately until convergence

    res_0 = 0
    tau = 0.5 + sigma**2*np.log(1/gamma-1)

    # calculate the sum of the first part
    l = 0
    terms_to_be_sum = []
    while True:
        # for numerical stability, we can only compute the log of it
        log_sum_term = log_generalized_binom_coeff(alpha, l)+(l-1)*l/2/sigma**2+(alpha-l)*np.log(1-gamma)+l*np.log(gamma)+np.log(erfc((l-tau)/np.sqrt(2)/sigma))
        terms_to_be_sum.append(log_sum_term)
        res_1 = log_sum_exp(np.array(terms_to_be_sum)) # calculate the log(eps) up to l steps using log-sum-exp
        change = np.abs((res_1 - res_0) / res_0) # the amount of change in log-scale after adding the l-th term
        res_0 = res_1
        l += 1
        if np.abs(change) <= conv_delta: # if the change is small, the sum converges
            break
    final_res_part_a = res_0 # get the sum result of the first part


    # calculate the sum of the second part
    res_0 = 0
    l = 0
    change_old = 0.
    terms_to_be_sum = []
    while True:
        log_sum_term = log_generalized_binom_coeff(alpha, l)+(alpha-l-1)*(alpha-l)/2/sigma**2+l*np.log(1-gamma)+(alpha-l)*np.log(gamma)+np.log(erfc((tau-alpha+l)/np.sqrt(2)/sigma))
        terms_to_be_sum.append(log_sum_term)
        res_1 = log_sum_exp(np.array(terms_to_be_sum))
        change_new = np.abs((res_1 - res_0) / res_0)
        l += 1
        if (change_new-change_old) < 0 and np.abs(change_new) <= conv_delta:
            # this sum first grows up and then decays, so to determine whether it converges, we also need to confirm the second derivative (difference) is negative
            break
        res_0 = res_1
        change_old = change_new

    final_res_part_b = res_0
    final_res_part_a, final_res_part_b = final_res_part_a - np.log(2), final_res_part_b - np.log(2) # divided by 2 in log scale
    res = log_sum_exp(np.array([final_res_part_a, final_res_part_b])) # get final results in log-scale using log-sum-exp
    return res

#def calculate_amp_rdp_epsilon_old(alpha, gamma, sigma, num_neg_pairs, num_nodes, num_edges, conv_delta, C_bound_function):
    # conv_delta: convergence criterion
    #gaussian_epsilon = gaussian_rdp_epsilon(sigma)
#    exp_amp_epsilon = 0.
#    for m in range(num_edges):
#        proba = binom.pmf(m, num_edges, gamma)
#        Gamma_m = 1-(1-gamma)**K*(1-m*num_neg_pairs/num_nodes)
#        exp_amp_epsilon_change = proba * C_bound_function(alpha, Gamma_m, sigma)
#        exp_amp_epsilon += exp_amp_epsilon_change
#        delta = np.abs(exp_amp_epsilon_change)
#        if m >= num_edges*gamma and delta <= conv_delta:
#            break
#    amp_epsilon = np.log(exp_amp_epsilon) / (alpha - 1)
#    return amp_epsilon



def calculate_pos_neg_rdp_epsilon(alpha, gamma, sigma, K, num_neg_pairs, num_nodes, num_edges, poisson_rdp_func=_compute_rdp, conv_delta=1e-6):
#def calculate_pos_neg_rdp_epsilon(alpha, gamma, sigma, K, num_neg_pairs, num_nodes, num_edges, poisson_rdp_func=calculate_poisson_rdp_epsilon, conv_delta=1e-6):
    # compute the (log) RDP epsilon of positive+negative subsampling, with positive sampling rate (gamma), gaussian noise (sigma),
    # a real (alpha), maximal node degree (K), number of negative samples per positive sample (num_neg_pairs), node size (num_nodes),
    # edge size (num_edges).
    # conv_delta: convergence criterion

    # we will do the expectation sum_m binom(num_edges, m) * exp(poisson_rdp(Gamma_m))
    # the summation is doing in log scale, so m-th term is log(binom(num_edges, m)) + poisson_rdp(Gamma_m)
    res_0 = 0
    terms_to_sum = []
    for m in range(num_edges):
        proba = binom.pmf(m, num_edges, gamma)
        Gamma_m = 1-(1-gamma)**K*(1-m*num_neg_pairs/num_nodes) # effective gamma
        #log_epsilon_change = np.log(proba) + poisson_rdp_func(alpha, Gamma_m, sigma) # log(binom(num_edges, m)) + poisson_rdp(Gamma_m)
        log_epsilon_change = np.log(proba) + poisson_rdp_func(Gamma_m, sigma, alpha) * (alpha - 1)
        terms_to_sum.append(log_epsilon_change)
        res_1 = log_sum_exp(np.array(terms_to_sum)) # log(sum_{k=1}^m binom(num_edges, k) * exp(poisson_rdp(Gamma_k)))
        change = np.abs((res_1 - res_0) / res_0) # used to determine convergence
        res_0 = res_1
        if m >= num_edges*gamma and change <= conv_delta:
            break
    epsilon = res_0 / (alpha - 1) # final eps
    return epsilon


#class NoiseCalibration:
#    def __init__(self, gamma, num_neg_pairs, num_nodes, num_edges, num_runs, adp_eps, adp_delta):
#        self.gamma = gamma
#        self.num_neg_pairs = num_neg_pairs
#        self.n = num_nodes
#        self.E = num_edges
#        self.num_runs = num_runs
#        self.adp_eps = adp_eps
#        self.adp_delta = adp_delta
#    def find_optimal_alpha(self, sigma):
#        adp_eps_func = lambda x: self.num_runs * calculate_amp_rdp_epsilon(x, self.gamma, sigma, self.num_neg_pairs, self.n, self.E, 1e-12, C_bound_real) + np.log(1/self.adp_delta) / (x-1)
#        res = minimize(adp_eps_func, np.array([10]), method='Nelder-Mead', bounds=[(2, None)])
#        opt_alpha, opt_eps = res.x.item(), res.fun
#        return opt_alpha, opt_eps
#    def noise_calibrate(self):
#        f = lambda x: (self.find_optimal_alpha(x)[1] - self.adp_eps)
#        sigma0 = brentq(f, 0.2, 2)
#        return sigma0
