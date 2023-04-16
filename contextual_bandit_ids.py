import numpy as np
import pandas as pd
from collections import defaultdict
from copy import copy
from scipy import stats
from helpers import *
from constants import *

from contextual_bandit import ContextualBandit


class ContextualBanditIDS(ContextualBandit):
    # variance based IDS (Alg 6 in Russo)
    # get v[arm] and delta[arm] for every arm
    def ids_helper(self, z, all_sample_thetas, all_sample_values):
        num_arms = all_sample_values.shape[0]
        num_samples = all_sample_values.shape[1]
        d = z.shape[0]
        mus = [self.posteriors[i].posterior_mu.ravel() for i in range(num_arms)]

        # vector of length `num_samples`
        max_values = all_sample_values.max(axis=0)
        # vector of length `num_samples`
        best_arm = all_sample_values.argmax(axis=0)
        expected_optimal = np.mean(max_values)

        # p_hat[arm] is percent that arm is optimal
        p_hat = dict()
        mu_hat_arm_i_when_j_opt = defaultdict(dict)
        for i in range(num_arms):
            for j in range(num_arms):
                indices_optimal = np.argwhere(best_arm == j).ravel()
                p_hat[j] = float(len(indices_optimal))/num_samples
                sample_thetas = all_sample_thetas[i]
                if len(indices_optimal) > 0:
                    mu_hat_arm_i_when_j_opt[i][j] = np.mean(sample_thetas[indices_optimal], axis=0)
                else:
                    # this case doesn't matter since p_hat is 0
                    mu_hat_arm_i_when_j_opt[i][j] = mus[i]

        L_hat_arm = dict()
        for i in range(num_arms):
            # sometimes L_hat is nan
            L_hat = np.zeros((d, d))
            for j in range(num_arms):
                diff = (mu_hat_arm_i_when_j_opt[i][j]-mus[i]).reshape(-1, 1) # column vec
                # import pdb; pdb.set_trace()
                L_hat += p_hat[j]*(diff @ diff.T)
            L_hat_arm[i] = L_hat

        # v_a
        v = dict()
        # delta_a: expected regret
        delta = dict()
        for i in range(num_arms):
            v[i] = (z.T @ L_hat_arm[i] @ z)[0][0]
            delta[i] = expected_optimal - (mus[i] @ z)[0]

        return v, delta


    def action(self, context, group=0):
        # sample
        all_sample_thetas = []
        # k by num_samples matrix
        all_sample_values = []
        for i in range(self.k):
            # num_samples by d matrix
            sample_thetas = self.posteriors[i].sample(num_samples=self.num_samples)
            # vector of length num_samples
            sample_values = (sample_thetas @ context).ravel()
            all_sample_thetas.append(sample_thetas)
            all_sample_values.append(sample_values)

        # calculate variance and delta on each arm
        v, d = self.ids_helper(context, all_sample_thetas, np.array(all_sample_values))
        best = None
        best_arms = None
        r = np.zeros(self.k)
        if len(v) == 1:
            best_arm = list(v.keys())[0]
            r[best_arm] = 1
            return r
        for arm in v.keys():
            for arm2 in v.keys():
                if arm != arm2:
                    p, val = ids_gradient_descent(v[arm], v[arm2], d[arm], d[arm2])
                    if best is None or val < best:
                        best = val
                        best_arms = arm, arm2, p
        arm, arm2, p = best_arms
        if self.t % 500 == 0 and self.t != 0:
            vals = []
            for i in range(self.k):
                vals.append(d[i]**2 / v[i])
                pr = p
            # print(np.argmin(vals), np.round(np.array(vals), 3))
            # import pdb; pdb.set_trace()
        if np.random.random() < p:
            r[arm] = 1
        else:
            r[arm2] = 1
        return r
