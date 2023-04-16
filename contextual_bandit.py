import numpy as np
import pandas as pd
from collections import defaultdict
from copy import copy
# from gurobipy import *
from scipy import stats
from helpers import *
from constants import *
# from linear_bandit import LinearBandit

from base_bandit import BaseBandit


def rd(num, dec=3):
    return np.round(num, dec)

class Posterior:
    def __init__(self, true_theta, prior_mu, prior_covariance,
                 noise_std, use_variance_estimate=False, unknown_noise=False):
        self.true_theta = true_theta
        self.d = len(self.true_theta.ravel())
        self.prior_mu = prior_mu
        self.prior_covariance = prior_covariance
        self.prior_inverse = np.linalg.inv(prior_covariance)
        self.prior_var_magnitude = self.prior_covariance[0][0]
        self.noise_std = noise_std
        self.ridge_lambda = (self.noise_std**2)/self.prior_var_magnitude
        self.V = np.eye(self.d)*self.ridge_lambda
        # self.V = self.prior_inverse
        self.X = None
        self.Y = None
        self.estimated_variance = 0
        self.use_variance_estimate = use_variance_estimate
        self.unknown_noise = unknown_noise

        self.posterior_mu = prior_mu
        self.posterior_covariance = prior_covariance

        self.ucb_alpha = 1

        self.a0 = 1
        self.b0 = 1
        self.a = 1
        self.b = 1
        self.delta = 0.0001

        self.true_approximation_error_std = 1
        self.estimate_noise = False
        self.freq = False

    def num_pulled(self):
        if self.X is None:
            return 0
        return self.X.shape[0]

    def update(self, context, reward):
        if self.X is None:
            self.X = np.array([context.ravel()])
            self.Y = np.vstack([reward])
        else:
            self.X = np.vstack([self.X, context.ravel()])
            self.Y = np.vstack([self.Y, reward])
        return self.do_update()

    def hypothetical_update_params(self, context, reward):
        if self.X is None:
            hyp_X = np.array([context.ravel()])
            hyp_Y = np.vstack([reward])
        else:
            hyp_X = np.vstack([self.X, context.ravel()])
            hyp_Y = np.vstack([self.Y, reward])
        num = hyp_X.shape[0]
        weight_matrix = np.diag(1/(np.ones(num)*self.noise_std**2))

        posterior_cov_inverse = self.prior_inverse + hyp_X.T @ weight_matrix  @ hyp_X
        posterior_covariance = np.linalg.inv(posterior_cov_inverse)
        posterior_mu = posterior_covariance @ hyp_X.T @ weight_matrix @ hyp_Y.reshape(-1, 1)
        return posterior_mu, posterior_covariance

    def update_true_approximation_error(self, std):
        self.true_approximation_error_std = std

    def do_update(self):
        num = self.X.shape[0]
        weight_matrix = np.diag(1/(np.ones(num)*self.noise_std**2))
        self.V = self.ridge_lambda*np.eye(self.d)+ self.X.T @ self.X
        self.posterior_cov_inverse = self.prior_inverse + self.X.T @ weight_matrix  @ self.X
        self.posterior_covariance = np.linalg.inv(self.posterior_cov_inverse)
        self.posterior_mu = self.posterior_covariance @ self.X.T @ weight_matrix @ self.Y.reshape(-1, 1)

        # update inverse gamma distribution
        self.a = self.a0 + num/2
        inverse_for_gamma = self.X.T @ self.X + self.prior_inverse
        mu_for_gamma = np.linalg.inv(inverse_for_gamma) @ self.X.T @ self.Y.reshape(-1, 1)
        self.b = self.b0 + (1/2)*( self.Y.T @ self.Y - mu_for_gamma.T @ inverse_for_gamma @ mu_for_gamma)[0][0]
        # print('gamma', self.a, self.b, self.b/(self.a-1))

        if self.unknown_noise:
            self.posterior_cov_inverse = inverse_for_gamma
            self.posterior_covariance = np.linalg.inv(self.posterior_cov_inverse)
            self.posterior_mu = mu_for_gamma

    def current_error(self, G_hat):
        beta_hat = (self.posterior_mu.T @  G_hat).ravel()
        # import pdb; pdb.set_trace()
        return np.linalg.norm(beta_hat- self.true_theta.ravel())

    def sample_noise(self):
        # https://en.wikipedia.org/wiki/Inverse-gamma_distribution
        var_sample = self.b * stats.invgamma.rvs(self.a)
        # if self.a > 1 and np.random.rand() < 0.001:
        #     print('noise:', np.round(var_sample, 3), np.sqrt(var_sample - self.noise_std**2), np.round(self.b/(self.a-1), 3))
        #     print('true noise std:', np.round(self.true_approximation_error_std, 3))

        if var_sample > self.noise_std**2:
            sigma = np.sqrt(var_sample - self.noise_std**2)
            return np.random.normal(0, sigma)
        return 0

    def sample_true_noise(self):
        return np.random.normal(0, self.true_approximation_error_std)

    def true_noise_ucb(self, t, a=1, b=1):
        alpha = a/(b+t)
        return stats.norm.ppf(1-alpha, scale=self.true_approximation_error_std)

    def sample(self, num_samples=1):
        if self.unknown_noise:
            var_sample = self.b * stats.invgamma.rvs(self.a)
            return np.random.multivariate_normal(self.posterior_mu.ravel(), var_sample*self.posterior_covariance).reshape(-1, 1)
        if num_samples == 1:
            return np.random.multivariate_normal(self.posterior_mu.ravel(), self.posterior_covariance).reshape(-1, 1)
        # each row is a sample
        return np.random.multivariate_normal(self.posterior_mu.ravel(), self.posterior_covariance, num_samples)

    def sample_rewards(self, context, num_samples=1):
        if num_samples == 1:
            return self.sample_reward(context)
        samples = self.sample(num_samples)
        return (samples @ context).ravel()

    def sample_reward(self, context):
        noise = 0
        if self.use_variance_estimate:
            if self.estimate_noise:
                noise = self.sample_noise()
            else:
                noise = self.sample_true_noise()
        sampled_theta = self.sample()
        # print(noise, sampled_theta, context, sampled_theta.T @ context)
        return noise + (sampled_theta.T @ context).ravel()[0]

    def greedy_reward(self, context):
        return (self.posterior_mu.ravel() @ context.ravel()).ravel()[0]

    def true_reward(self, context):
        return (self.true_theta.T @ context).ravel()[0]

    def estimated_sigma(self, num_params):
        n, d = self.X.shape
        Y_hat = (self.X @ self.posterior_mu).ravel()
        self.estimated_variance = np.sum((Y_hat - self.Y.ravel())**2)/(n - num_params)
        return self.estimated_variance

    def bayes_confidence_radius(self, t, a=1, b=1):
        deg_freedom = self.d
        alpha = a/(b+t)
        return np.sqrt(stats.chi2.ppf(1-alpha, deg_freedom))

    def freq_confidence_radius(self):
        det_V = np.linalg.det(self.V)
        det_lamb = np.linalg.det(np.identity(self.d)*self.ridge_lambda)
        S = 1
        return self.noise_std * np.sqrt(2*np.log(det_V**(1/2) / (self.delta*det_lamb**(1/2)))) + self.ridge_lambda**(1/2)*S

    def freq_ucb(self, context, t):
        # print('doing freq')
        beta = self.freq_confidence_radius()
        # if t % 500 == 0:
        #     bayes_beta = self.bayes_confidence_radius(t)
        #     norm = np.sqrt(context.T @ np.linalg.inv(self.V) @ context).ravel()[0]
        #     bayes_norm = np.sqrt(context.T @ self.posterior_covariance @ context).ravel()[0]
        #     print('freq norm', rd(norm), rd(bayes_norm), norm/bayes_norm)
        return (context.T @ self.posterior_mu + self.ucb_alpha * beta * np.sqrt(context.T @ np.linalg.inv(self.V) @ context)).ravel()[0]

    def ucb(self, context, t):
        if self.freq:
            return self.freq_ucb(context, t)
        beta = self.bayes_confidence_radius(t)
        # return context.T @ self.posterior_mu + beta * np.sqrt(context.T @ cov_inverse @ context)
        return (context.T @ self.posterior_mu + self.ucb_alpha * beta * np.sqrt(context.T @ self.posterior_covariance @ context)).ravel()[0]

class History:
    def __init__(self, arm, expected_reward, reward_with_noise,
                 optimal_reward, reward_so_far, optimal_arm, group=0,
                 delta_error=-1, **kwargs):
        self.arm = arm
        self.group = group
        self.optimal_arm = optimal_arm
        self.expected_reward = expected_reward
        self.reward_with_noise = reward_with_noise
        self.optimal_reward = optimal_reward
        self.reward_so_far = reward_so_far
        self.delta_error = delta_error
        self.__dict__.update(kwargs)
        self.instant_regret = self.optimal_reward - self.expected_reward

def get_stats_for_history(histories):
    reward_received = sum([h.expected_reward for h in histories])
    optimal_reward = sum([h.optimal_reward for h in histories])
    total_regret = sum([h.instant_regret for h in histories])
    greedy_reward = np.sum([h.greedy_reward for h in histories])
    prior_expected_reward = np.sum([h.prior_expected_reward for h in histories])
    return reward_received, optimal_reward, total_regret, greedy_reward, prior_expected_reward

class ContextualBandit(BaseBandit):
    def __init__(self, contexts, true_betas, prior_mu,
                 prior_var_magnitude, noise_std, T, groups=None):
        """
        true_betas is a K by d matrix
        """
        self.true_betas = true_betas
        self.k, self.d = true_betas.shape
        self.T = T

        # vector
        self.prior_mu = prior_mu.reshape(-1, 1)
        # scalar
        self.prior_var_magnitude = prior_var_magnitude
        self.prior_covariance = np.identity(self.d)*prior_var_magnitude
        self.noise_std = noise_std

        self.initialize = False
        self.posteriors = [Posterior(true_betas[i].reshape(-1, 1), self.prior_mu, self.prior_covariance, noise_std) for i in range(self.k)]

        self.t = 1
        self.history = []
        self.reward = 0
        self.regret = 0

        self.contexts = contexts
        self.groups = groups

        self.freq = False
        self.num_samples = 1

    def set_freq(self, freq):
        for p in self.posteriors:
            p.freq = freq

    def set_ucb_alpha(self, alpha):
        for p in self.posteriors:
            p.ucb_alpha = alpha

    # def set_estimate_noise(self):
    #     for p in self.posteriors:
    #         p.estimate_noise = True

    def regret_over_time(self):
        r = 0
        regrets = []
        for h in self.history:
            r += h.instant_regret
            regrets.append(r)
        return regrets

    def print_arms_pulled(self):
        arms = np.array([h.arm for h in self.history])
        print('arms pulled:')
        print(pd.value_counts(arms))
        return arms
        # print arms

    def get_history_values(self, attrs, hs):
        attrs_list = dict()
        for attr in attrs:
            attrs_list[attr] = np.array([getattr(h, attr) for h in hs])
        return attrs_list

    def get_history_cum_values(self, attrs, hs):
        summed_attrs_list = dict()
        for attr in attrs:
            l = [getattr(h, attr) for h in hs]
            summed_attrs_list[attr] = np.array(pd.Series(l).cumsum())
        return  summed_attrs_list

    def get_results(self, base_result, attrs, attrs_to_sum, intervals=100):
        attrs_list = self.get_history_values(attrs, self.history)
        summed_attrs_list = self.get_history_cum_values(attrs_to_sum, self.history)
        results = []
        for t in range(intervals, self.T+1, intervals):
            r = copy(base_result)
            r['time'] = t
            r['last_time_step'] = int(t + intervals > self.T)
            # import pdb; pdb.set_trace()
            for attr in attrs:
                r[attr] = attrs_list[attr][t]
            for attr in attrs_to_sum:
                r[attr + '_sum'] = summed_attrs_list[attr][t]
            results.append(r)
        return pd.DataFrame(results)

    def get_grouped_results(self, base_result, attrs,
        attrs_to_sum, grouped_attrs, intervals=100):
        attrs_list = self.get_history_values(attrs, self.history)
        summed_attrs_list = self.get_history_cum_values(attrs_to_sum, self.history)

        # grouped_histories = defaultdict(list)
        # for i, h in enumerate(self.history):
        #     grouped_histories[h.group].append(h)
        # grouped_attrs_list = dict()
        # for grp_name, hists in grouped_histories.items():
        #     grouped_attrs_list[grp_name] = self.get_history_cum_values(grouped_attrs, hists)

        results = []
        grouped_histories = defaultdict(list)
        for t, h in enumerate(self.history):
            grouped_histories[h.group].append(h)
            if t % intervals == 0 and t != 0:
                # print('adding!', t)
                r = copy(base_result)
                r['time'] = t
                r['last_time_step'] = int(t + intervals > self.T)
                for attr in attrs:
                    r[attr] = attrs_list[attr][t]
                for attr in attrs_to_sum:
                    r[attr + '_sum'] = summed_attrs_list[attr][t]

                for grp_name, hists in grouped_histories.items():
                    group_attrs_list = self.get_history_cum_values(grouped_attrs, grouped_histories[grp_name])
                    for attr in grouped_attrs:
                        r[attr + '_sum_g%s' % str(grp_name)] = group_attrs_list[attr][-1]
                        r['num_in_group_g%s' % str(grp_name)] = len(hists)
                results.append(r)
        return pd.DataFrame(results)

    def add_regret(self, results):
        i = 1
        regrets = self.regret_over_time()
        while i * 500 <= len(regrets):
            d = i*500
            results['%d_cum_regret' % d] = regrets[d-1]
            i += 1
        total_regret = 0
        grouped_histories = defaultdict(list)
        hists_so_far = []
        for t, h in enumerate(self.history):
            grouped_histories[h.group].append(h)
            hists_so_far.append(h)
            if (t % 500 == 0 and t != 0) or t == len(self.history)-1:
                _, _, total_regret, _, _ =  get_stats_for_history(hists_so_far)
                for grp_name, hists in grouped_histories.items():
                    g_reward, g_optimal_reward, g_regret, greedy_reward, prior_expected_reward = get_stats_for_history(hists)
                    time = 'total' if t == len(self.history) - 1 else str(t)
                    results['%s_reward_frac_g%s' % (time, str(grp_name))] = g_reward / g_optimal_reward
                    results['%s_regret_frac_g%s' % (time, str(grp_name))] = g_regret / total_regret
                    results['%s_num_frac_g%s' % (time, str(grp_name))] = len(hists) / len(hists_so_far)
                    results['%s_greedy_reward_g%s' % (time, str(grp_name))] = greedy_reward
                    results['%s_prior_expected_reward_g%s' % (time, str(grp_name))] = prior_expected_reward

        results['total_regret'] = self.regret
        results['total_reward'] = self.reward

    def action(self, context, group=0):
        # print('samples:', [p.sample_reward(context) for p in self.posteriors])
        return [p.sample_reward(context) for p in self.posteriors]

    def update(self, arm, context, reward):
        self.posteriors[arm].update(context, reward)
        if self.t % 100 == 10:
            beta_hat = np.array([p.posterior_mu.ravel() for p in self.posteriors])
            total_error = np.linalg.norm(beta_hat - self.true_betas)
            # print('total error', self.t, np.round(total_error, 3))

    @staticmethod
    def generate_context(d):
        context = np.random.normal(0,1/np.sqrt(d),size=d)
        return context.reshape(-1, 1)

    @staticmethod
    def generate_contexts(d, T):
        return [ContextualBandit.generate_context(d) for _ in range(T+10)]

    @staticmethod
    def generate_contexts_with_groups(d, T, p_small_group=0.1, small_magnitude=0.1):
        diag1 = np.ones(d)
        diag2 = np.ones(d)
        # diag1[:int(d/2)] = small_magnitude
        diag1[int(d-d/3):] = small_magnitude
        diag2[:int(d/3)] = small_magnitude
        rand = np.random.rand(T+10)
        contexts = []
        groups = []
        for i in range(T+10):
            # group is 0 by default, 1 if minority.
            group = 0
            if rand[i] < p_small_group:
                context = np.random.multivariate_normal(np.zeros(d),np.diag(diag2)/d)
                group = 1
            else:
                context = np.random.multivariate_normal(np.zeros(d),np.diag(diag1)/d)

            # # normalize
            # context = (context/np.linalg.norm(context)).reshape(-1, 1)

            contexts.append(context.reshape(-1, 1))
            groups.append(group)
        return contexts, groups

    @staticmethod
    def generate_nondiverse_contexts(d, T, epoch_length=300, num_different=3):
        if epoch_length == 1:
            return [ContextualBandit.generate_context(d) for _ in range(T+10)]
        contexts = []
        while len(contexts) <= T:
            current_contexts = [ContextualBandit.generate_context(d) for _ in range(num_different)]
            random_ordering_idx = np.random.choice(range(num_different), size=epoch_length)
            contexts.extend([current_contexts[i] for i in random_ordering_idx])
        return contexts

    def pull(self, arm, context, group=0):
        expected_reward = self.posteriors[arm].true_reward(context)
        reward_with_noise = np.random.normal(expected_reward, self.noise_std)

        # Given the knowledge we have so far, how good is the arm
        # compared to greedy?
        # Need to do this before we update posteriors.
        prior_expected_rewards = [p.greedy_reward(context) for p in self.posteriors]
        greedy_reward = np.max(prior_expected_rewards)
        prior_expected_reward = prior_expected_rewards[arm]

        # if group == 1:
        #     print('\ngreedy_reward', np.round(greedy_reward, 3))
        #     print('prior_expected_reward', np.round(prior_expected_reward, 3))

        self.update(arm, context, reward_with_noise)
        optimal_reward = np.max([p.true_reward(context) for p in self.posteriors])
        optimal_arm = np.argmax([p.true_reward(context) for p in self.posteriors])



        self.reward += expected_reward
        self.regret += (optimal_reward - expected_reward)
        self.history.append(History(arm, expected_reward, reward_with_noise,
                                    optimal_reward, self.reward, optimal_arm, group,
                                    greedy_reward=greedy_reward,
                                    prior_expected_reward=prior_expected_reward))
        self.t += 1

    def run_contextual(self, initialize=False, **kwargs):
        while self.t <= self.T+1:
            context = self.contexts[self.t]
            group = 0 if self.groups is None else self.groups[self.t]
            arm = np.argmax(self.action(context, group=group, **kwargs))
            if False and self.t % 10 == 0:
            # if group == 1:
                print()
                print(self.t)
                self.log_hypothetical(context)
            self.pull(arm, context, group)
            # if self.t % 500 == 0:
            #     print('\nupdating', self.t)
            #     print('regret so far', self.regret)
            #     print('reward so far', self.reward)
            #     self.print_arms_pulled()

    def sample_optimal(self, context, num_samples=1):
        rewards = np.array([p.sample_rewards(context, num_samples) for p in self.posteriors])
        # import pdb; pdb.set_trace()
        return np.mean(np.max(rewards, axis=0))
        # sampled_optimals = [np.max([p.sample_reward(context) for p in self.posteriors]) for _ in range(num_samples)]
        # return np.mean(sampled_optimals)

    def L_increase(self, context, num_samples=100):
        # each row represents one arm
        samples = np.array([p.sample_rewards(context, num_samples) for p in self.posteriors])
        expected_optimal = np.mean(np.max(samples, axis=0))
        greedy_reward = np.max([p.greedy_reward(context) for p in self.posteriors])

        curr_L = expected_optimal - greedy_reward
        new_Ls = []
        print()
        for k, p in enumerate(self.posteriors):
            arm_samples = samples[k]
            new_L_samples = []
            for s in arm_samples:
                updated_mu, updated_cov = p.hypothetical_update_params(context, s)
                new_sampled_param = np.random.multivariate_normal(
                    updated_mu.ravel(), updated_cov, num_samples)
                new_samples = (new_sampled_param @ context).ravel()

                new_samples = np.vstack((samples[:k, :], samples[k+1:,], new_samples))
                new_expected_optimal = np.mean(np.max(samples, axis=0))

                arm_greedy = updated_mu.ravel() @ context.ravel()
                new_greedy = max(greedy_reward, arm_greedy)

                new_L  = new_expected_optimal - new_greedy
                new_L_samples.append(new_L)
            new_Ls.append(np.mean(new_L_samples))
            print(k, rd(p.greedy_reward(context)), rd(np.mean(new_L_samples)))
        print(self.t, 'arm:', np.argmin(new_Ls))
        # if self.t % 50 == 0:
        #     print('\nupdating', self.t)
        #     print('regret so far', self.regret)
        #     print('reward so far', self.reward)
        return -np.array(new_Ls)

    def log_hypothetical(self, context):
        sampled_optimal = self.sample_optimal(context)
        greedys = [p.greedy_reward(context) for p in self.posteriors]
        ts_samples = [p.sample_reward(context) for p in self.posteriors]
        ucbs = [p.ucb(context, self.t) for p in self.posteriors]
        ratios = []
        for i, p in enumerate(self.posteriors):
            mean = greedys[i]
            ucb = ucbs[i]
            ts = ts_samples[i]
            ratio = (sampled_optimal - mean)/(ucb-mean)
            ratios.append(ratio)
            print(i, rd(mean), rd(ts), rd(ucb), rd(ratio))
        print('greedy:', np.argmax(greedys))
        print('ts:', np.argmax(ts_samples))
        print('ucb:', np.argmax(ucbs))
        print('tsucb:', np.argmin(ratios))


class ContextualBanditUCB(ContextualBandit):
    def action(self, context, group=0):
        return [p.ucb(context, self.t) for p in self.posteriors]

class ContextualBanditTSUCB(ContextualBandit):
    def action(self, context, group=0):
        ratios = []
        sampled_optimal = self.sample_optimal(context, num_samples=self.num_samples)
        for p in self.posteriors:
            mean = p.greedy_reward(context)
            ucb = p.ucb(context, self.t)
            # negative sign since we want to minimize the ratio
            ratios.append(-(sampled_optimal - mean)/(ucb-mean))
        return ratios

class ContextualBanditGreedy(ContextualBandit):
    def action(self, context, group=0):
        # print([p.greedy_reward(context) for p in self.posteriors])
        return [p.greedy_reward(context) for p in self.posteriors]


