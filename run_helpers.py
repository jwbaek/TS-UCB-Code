import numpy as np
import pandas as pd
from collections import defaultdict
from copy import copy
from scipy.stats import norm
import time
# from gurobipy import *
from scipy import stats
import pickle
from linear_bandit import *
from contextual_bandit import ContextualBandit
from mab import *

from sklearn.preprocessing import normalize

def rename(alg):
    replacements = [
    ('exp_', ''),
    ('tcucb', 'tsucb'),
    ('ts_then_ucb', 'TSUCB'),
    ('match_ts_ub_lp_one_sample', 'TSUCB_optimize_1sample'),
    ('match_ts_ub_lp', 'TSUCB_optimize'),
    ('tsucb', 'TSUCB'),
    ('ts_then_ids', 'TSIDS'),
    ('ts', 'TS'),
    ('ids', 'IDS') ]
    l = alg
    for to_replace, replace in replacements:
        l = l.replace(to_replace, replace)
    return l

def do_one_instance_karm(k, T, prior_mu, prior_std, noise_std, seed, alg, num_iters, alg_name=None, **kwargs):
    if alg_name is None:
        alg_name = alg
    out = {
        'k':k,
        'T':T,
        'seed': seed,
        'alg': alg_name,
       }
    np.random.seed(seed)
    true_mu = np.random.normal(prior_mu, prior_std)

    bandit = Karm(true_mu, prior_mu, prior_std, noise_std, T)
    start = time.time()

    regrets = []
    for _ in range(num_iters):
        bandit.run(alg, initialize=True, **kwargs)
        regrets.append(bandit.instant_regrets())

    out['time'] = (time.time() - start)/num_iters
    avg_regret = np.mean(np.array(regrets), axis=0)
    prev = 0
    for t, r in enumerate(avg_regret):
        prev += r
        out['cum_regret_%d' % t] = prev
    out['total_regret'] = prev
    return out, bandit

# This is the function that we used to run experiments on engaging for linear bandit.
def real_do_one_instance(d, k, T, prior_std_magnitude, seed, algs, noise_std=1, noise_variances=None, return_bandit=False, **kwargs):
    np.random.seed(seed)
    prior_mu = np.zeros(d)
    true_theta = np.random.multivariate_normal(prior_mu, np.identity(d)*prior_std_magnitude)
    actions = np.random.random((k, d)) - 0.5
    # all actions have norm 1
    actions = normalize(actions, axis=1)

    lb = LinearBandit(true_theta, actions, prior_mu, np.sqrt(prior_std_magnitude), noise_std, T, noise_variances=noise_variances)
    start = time.time()

    all_results = []
    for alg in algs:
        start_time = time.time()
        lb.run(alg, **kwargs)
        time_taken = time.time() - start_time
        results = {'d': d,
        'time_taken': time_taken,
        'k':k,
        'T':T,
        'prior_std_magnitude': prior_std_magnitude,
        'seed': seed,
        'alg': alg
        }
        results.update(kwargs)
        ins_regrets = lb.instant_regrets()
        prev = 0
        for t, r in enumerate(ins_regrets):
            prev += r
            # write only every 10 time steps
            if t %10 == 0:
                results['cum_regret_%d' % t] = prev
        results['total_regret'] = np.sum(ins_regrets)
        results['total_reward'] = lb.reward
        results['optimal_reward'] = lb.optimal_reward*T
        all_results.append(results)

    if return_bandit:
        return lb, all_results

    return all_results

def real_do_one_instance_karm(k, T, kappa, seed, algs, noise_std=1, **kwargs):
    # k = num arms
    prior_mu = np.zeros(k)
    prior_std = np.ones(k) * kappa

    np.random.seed(seed)
    true_mu = np.random.normal(prior_mu, prior_std)

    bandit = Karm(true_mu, prior_mu, prior_std, noise_std, T)
    start = time.time()

    all_results = []
    for alg in algs:
        start_time = time.time()
        bandit.run(alg, initialize=True, **kwargs)
        time_taken = time.time() - start_time
        results = {
        'k':k,
        'T':T,
        'seed': seed,
        'kappa': kappa,
        'time_taken': time_taken,
        'alg': alg, }

        results.update(kwargs)
        ins_regrets = bandit.instant_regrets()
        prev = 0
        for t, r in enumerate(ins_regrets):
            prev += r
            # write only every 10 time steps
            if t %10 == 0:
                results['cum_regret_%d' % t] = prev

        results['total_regret'] = np.sum(ins_regrets)
        results['total_reward'] = bandit.reward
        results['optimal_reward'] = bandit.optimal_reward*T
        all_results.append(results)
    return all_results


def do_one_instance(d, k, T, prior_mu, prior_std, prior_std_magnitude, noise_std,
                    seed, alg, num_iters, alg_name=None, arm_hetero_range=0, return_bandit=False, **kwargs):
    if alg_name is None:
        alg_name = alg
    out = {'d': d,
        'k':k,
        'T':T,
        'prior_std_magnitude': prior_std_magnitude,
        'num_iters_per_instance': num_iters,
        'seed': seed,
        'alg': alg_name,
        'arm_hetero_range': arm_hetero_range,
        }
    np.random.seed(seed)

    true_theta = np.random.multivariate_normal(prior_mu, prior_std)
    # each component is uniform around -1/sqrt{10} and 1/sqrt{10}
    actions = 2*np.random.random((k, d))/np.sqrt(d) - 1/np.sqrt(d)
    noise_variances =  np.random.rand(k)*arm_hetero_range + 1 - arm_hetero_range/2
    lb = LinearBandit(true_theta, actions, prior_mu, prior_std_magnitude, noise_std, T, arm_noise_variances=noise_variances)
    start = time.time()

    regrets = []
    for _ in range(num_iters):
        lb.run(alg, **kwargs)
        regrets.append(lb.instant_regrets())

    out['time'] = (time.time() - start)/num_iters
    avg_regret = np.mean(np.array(regrets), axis=0)
    prev = 0
    for t, r in enumerate(avg_regret):
        prev += r
        out['cum_regret_%d' % t] = prev
    out['total_regret'] = prev

    if return_bandit:
        return out, lb

    return out


def do_one_contextual_instance(d, k, T, prior_mu, prior_std, prior_std_magnitude, noise_std,
                    seed, alg, num_iters, alg_name=None, arm_hetero_range=0, action_err_std=0,
                    return_bandit=False, action_stds=None, **kwargs):
    if alg_name is None:
        alg_name = alg
    out = {'d': d,
        'k':k,
        'T':T,
        'prior_std_magnitude': prior_std_magnitude,
        'num_iters_per_instance': num_iters,
        'seed': seed,
        'alg': alg_name,
        'arm_hetero_range': arm_hetero_range,
        }
    np.random.seed(seed)

    true_theta = np.random.multivariate_normal(prior_mu, prior_std)
    # each component is uniform around -1/sqrt{10} and 1/sqrt{10}
    actions = 2*np.random.random((k, d))/np.sqrt(d) - 1/np.sqrt(d)
    lb = ContextualBandit(true_theta, actions, prior_mu, prior_std_magnitude, noise_std, T, action_err_std=action_err_std, action_stds=action_stds)
    start = time.time()

    regrets = []
    for _ in range(num_iters):
        lb.run(alg, **kwargs)
        regrets.append(lb.instant_regrets())

    out['time'] = (time.time() - start)/num_iters
    avg_regret = np.mean(np.array(regrets), axis=0)
    prev = 0
    for t, r in enumerate(avg_regret):
        prev += r
        out['cum_regret_%d' % t] = prev
    out['total_regret'] = prev

    if return_bandit:
        return out, lb

    return out

