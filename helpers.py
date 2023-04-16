# from gurobipy import *
import numpy as np
from constants import *
import scipy

from sklearn.preprocessing import normalize

# def create_new_actions(d, k):
#     actions = np.random.random((k, d)) - 0.5
#     # all actions have norm 1
#     actions = normalize(actions, axis=1)
#     return actions

def ids_gradient_descent(v1, v2, d1, d2, num_iters=4):
    F = lambda p: (p*d1 + (1-p)*d2)**2/(p*v1 + (1-p)*v2)
    lower_p = 0
    upper_p = 1

    for t in range(num_iters):
        current_ps = np.arange(lower_p, upper_p + EPS, (upper_p - lower_p)/10)
        fs = [F(p) for p in current_ps]
        argmin = np.argmin(fs)
        lower_p = current_ps[argmin] if argmin == 0 else current_ps[argmin-1]
        upper_p = current_ps[argmin] if argmin == len(fs)-1 else current_ps[argmin+1]

    opt_p = current_ps[argmin]

    return opt_p, F(opt_p)

def ids_gradient_descent_old(v1, v2, d1, d2, num_iters=4):
    F = lambda p: (p*d1 + (1-p)*d2)**2/(p*v1 + (1-p)*v2)
    lower_p = 0
    upper_p = 1

    opt_p = None
    for t in range(num_iters):
        current_ps = np.arange(lower_p, upper_p + EPS, (upper_p - lower_p)/10)
        fs = [F(p) for p in current_ps]
        m = np.min(fs)
        argmin = np.argmin(fs)
        if argmin == 0:
            opt_p = current_ps[0]
            break
        if argmin == len(fs)-1:
            opt_p = current_ps[-1]
            break
        lower_p = current_ps[argmin-1]
        upper_p = current_ps[argmin+1]

    opt_p = current_ps[argmin]

    return opt_p, F(opt_p)

def opt_gittins_search(samples, mu, ub, gamma):
    mu = np.mean(samples)
    low_lamb = mu
    high_lamb = ub
    f = lambda lamb  : mu - lamb + gamma*np.mean(np.maximum(lamb - samples, 0))
    while f(high_lamb) > EPS:
        high_lamb += (ub-mu)
    while high_lamb - low_lamb > 0.01:
        curr_lamb = (high_lamb + low_lamb)/2
        rhs = mu + gamma*np.mean(np.maximum(curr_lamb - samples, 0))
        if curr_lamb > rhs:
            # curr lamb was too high
            high_lamb = curr_lamb
        else:
            low_lamb = rhs
    return low_lamb

def opt_gittins_search2(samples, mu, ub, gamma):
    mu = np.mean(samples)
    assert ub >= mu
    low_lamb = mu
    high_lamb = ub
    f = lambda lamb  : mu - lamb + gamma*np.mean(np.maximum(lamb - samples, 0))
    while f(high_lamb) > 0:
        high_lamb += (ub-mu)
    return scipy.optimize.bisect(f, low_lamb, high_lamb)

def ucb_gittins_search(samples):
    for s in samples:
        pass

def optimize_p(mus, ucbs, expected_ts):
    """
    Maximize greedy reward while expected ucb is exactly expected_ts
    """
    k = len(ucbs)
    m = Model()
    m.setParam('OutputFlag', False)
    m.modelSense = GRB.MAXIMIZE

    if np.max(ucbs) < expected_ts - 0.01 or np.min(ucbs) > expected_ts + 0.01:
        print('assertion:', m.status, expected_ts, np.min(ucbs), np.max(ucbs), ucbs)

    ps = dict()
    arms = list(range(len(mus)))
    for arm in arms:
        ps[arm] = m.addVar(name='p[%d]' % arm, lb=0, ub=1, obj=mus[arm])
    m.addConstr(sum(ps[arm] for arm in arms) == 1)

    m.addConstr(sum(ps[arm]*ucbs[arm] for arm in arms) <= expected_ts + 0.005)
    m.addConstr(sum(ps[arm]*ucbs[arm] for arm in arms) >= expected_ts)

    m.optimize()

    if m.status != 2:
        print('Optimal not found:', m.status, expected_ts, np.min(ucbs), np.max(ucbs), ucbs)

    p = dict()
    for arm, v in ps.items():
        p[arm] = v.x
    return p


def optimize_p2(mus, ucbs, expected_ts):
    """
    Maximize greedy reward - expected ucb while expected ucb >= expected_ts
    """
    k = len(ucbs)
    m = Model()
    m.setParam('OutputFlag', False)
    m.modelSense = GRB.MAXIMIZE

    if np.max(ucbs) < expected_ts - 0.01 or np.min(ucbs) > expected_ts + 0.01:
        print('assertion:', m.status, expected_ts, np.min(ucbs), np.max(ucbs), ucbs)

    ps = dict()
    arms = list(range(len(mus)))
    for arm in arms:
        ps[arm] = m.addVar(name='p[%d]' % arm, lb=0, ub=1, obj=mus[arm] - ucbs[arm])
    m.addConstr(sum(ps[arm] for arm in arms) == 1)

    # m.addConstr(sum(ps[arm]*ucbs[arm] for arm in arms) <= expected_ts + 0.005)
    m.addConstr(sum(ps[arm]*ucbs[arm] for arm in arms) >= expected_ts)

    m.optimize()

    if m.status != 2:
        print('Optimal not found:', m.status, expected_ts, np.min(ucbs), np.max(ucbs), ucbs)

    p = dict()
    for arm, v in ps.items():
        p[arm] = v.x
    return p
