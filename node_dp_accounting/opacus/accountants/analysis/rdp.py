#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

r"""
*Based on Google's TF Privacy:* https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py.
*Here, we update this code to Python 3, and optimize dependencies.*

Functionality for computing Renyi Differential Privacy (RDP) of an additive
Sampled Gaussian Mechanism (SGM).

Example:
    Suppose that we have run an SGM applied to a function with L2-sensitivity of 1.

    Its parameters are given as a list of tuples
    ``[(q_1, sigma_1, steps_1), ..., (q_k, sigma_k, steps_k)],``
    and we wish to compute epsilon for a given target delta.

    The example code would be:

    >>> parameters = [(1e-5, 1.0, 10), (1e-4, 3.0, 4)]
    >>> delta = 1e-5

    >>> max_order = 32
    >>> orders = range(2, max_order + 1)
    >>> rdp = np.zeros_like(orders, dtype=float)
    >>> for q, sigma, steps in parameters:
    ...     rdp += compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)

    >>> epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp, delta=1e-5)
    >>> float(epsilon), int(opt_order)  # doctest: +NUMBER
    (0.336, 23)

"""

import math
import warnings
from typing import List, Tuple, Union
from tqdm import tqdm
import numpy as np
from scipy import special
from scipy.stats import binom

from group_amplification.privacy_analysis.base_mechanisms import GaussianMechanism
from group_amplification.privacy_analysis.subsampling.rdp.poisson import _rdp_tight_pos_neg_quadrature_gaussian

########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    r"""Computes :math:`log(A_\alpha)` for integer ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )

        s = log_coef_i + (i * i - i) / (2 * (sigma**2))
        log_a = _log_add(log_a, s)

    return float(log_a)


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma**2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for any positive finite ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf
        for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in the paper mentioned above.
    """
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.

    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.

    Args:
        x: The input to the function

    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2**0.5)


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    r"""Computes RDP of the Sampled Gaussian Mechanism at order ``alpha``.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    if q == 1.0:
        return alpha / (2 * sigma**2)

    if np.isinf(alpha):
        return np.inf

    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(
    *, q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SGM.
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.

    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps



def log_sum_exp(x):
    # x is an array, return log(exp(x_1)+...+exp(x_n))
    xmax = x.max()
    if xmax == -np.inf: # in case all x are -inf
        return -np.inf
    else:
        return np.log(sum(np.exp(x-xmax))) + xmax # a numerically stable way


def compute_pos_neg_rdp(q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float], max_node_degree: int,
                        num_neg_pairs: int, num_nodes: int, num_edges: int, conv_criterion=1e-4, constant_sensitivity=True):
        # def calculate_pos_neg_rdp_epsilon(alpha, gamma, sigma, K, num_neg_pairs, num_nodes, num_edges, poisson_rdp_func=calculate_poisson_rdp_epsilon, conv_delta=1e-6):
        # compute the (log) RDP epsilon of positive+negative subsampling, with positive sampling rate (gamma), gaussian noise (sigma),
        # a real (alpha), maximal node degree (K), number of negative samples per positive sample (num_neg_pairs), node size (num_nodes),
        # edge size (num_edges).
        # conv_delta: convergence criterion

        # we will do the expectation sum_m binom(num_edges, m) * exp(poisson_rdp(Gamma_m))
        # the summation is doing in log scale, so m-th term is log(binom(num_edges, m)) + poisson_rdp(Gamma_m)
        eps_all_orders = []
        orders = [orders] if isinstance(orders, float) else orders
        for alpha in tqdm(orders, desc='Computing RDP bound at different orders alpha'):
            res_0 = 0
            terms_to_sum = []
            mean, std = num_edges * q, np.sqrt(num_edges * q * (1 - q))
            m_start = int(max(0, mean - 6 * std))
            m_end = int(mean + 6 * std)
            temp = []
            #for m in range(num_edges):
            for m in tqdm(range(m_start, m_end), desc="At alpha=%.2f, iter over all batch size" % alpha, leave=False):
                proba = binom.pmf(m, num_edges, q)
                if constant_sensitivity:
                    # when the sensitivity is constant, the RDP can be reduced to a non-group privacy problem with an effective sampling rate
                    Gamma_m = 1 - (1 - q) ** max_node_degree * (1 - m * num_neg_pairs / num_nodes)  # effective gamma
                    log_epsilon_change = np.log(proba) + _compute_rdp(Gamma_m, noise_multiplier, alpha) * (alpha - 1)
                    # log_epsilon_change = np.log(proba) + poisson_rdp_func(alpha, Gamma_m, sigma) # log(binom(num_edges, m)) + poisson_rdp(Gamma_m)
                else:
                    # when sensitivity varies, directly use numerical integral for the group privacy RDP
                    rdp=_rdp_tight_pos_neg_quadrature_gaussian(alpha, GaussianMechanism(noise_multiplier), q, 0,
                                                           max_node_degree, 0, m * num_neg_pairs / num_nodes,
                                                           {'dps':10}, constant_sensitivity=False)
                    #B=_rdp_tight_pos_neg_quadrature_gaussian(alpha, GaussianMechanism(noise_multiplier), q,
                                                           #max_node_degree, 0, m * num_neg_pairs / num_nodes, 0,
                                                           #{'dps':50}, constant_sensitivity=False)
                    log_epsilon_change = np.log(proba) + rdp * (alpha - 1)
                    temp.append(rdp)
                terms_to_sum.append(log_epsilon_change)
                res_1 = log_sum_exp(
                    np.array(terms_to_sum))  # log(sum_{k=1}^m binom(num_edges, k) * exp(poisson_rdp(Gamma_k)))
                change = np.abs((res_1 - res_0) / np.max([np.abs(res_0), np.abs(res_1)]))  # used to determine convergence
                res_0 = res_1
                if m >= num_edges * q and change <= conv_criterion:
                    break
            epsilon = res_0 / (alpha - 1) * steps  # final eps
            eps_all_orders.append(epsilon)
        return np.array(eps_all_orders)




def compute_pos_neg_lower(q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float], max_node_degree: int,
                        num_neg_pairs: int, num_nodes: int, num_edges: int, conv_criterion=1e-6):
    eps_all_orders = []
    orders = [orders] if isinstance(orders, float) else orders
    for alpha in orders:
        gamma_b = 0
        gamma_effective = 0
        for b in range(1, num_edges - max_node_degree + 1):
            gamma_b = 1 - (1 - gamma_b) * (num_edges - max_node_degree - b + 1) / (num_edges - b + 1) * (
                        num_nodes - b * num_neg_pairs) / (num_nodes - b * num_neg_pairs + 1)
            proba_b = binom.pmf(b, num_edges, q)
            gamma_effective += proba_b * gamma_b
            if b >= 2 * num_edges * q and (proba_b * gamma_b / gamma_effective) <= conv_criterion:
                break
        res = _compute_rdp(gamma_effective, noise_multiplier, alpha) * steps
        eps_all_orders.append(res)
    return np.array(eps_all_orders)


def get_privacy_spent(
    *, orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
        )
    return eps[idx_opt], orders_vec[idx_opt]
