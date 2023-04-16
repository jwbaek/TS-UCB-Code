import click
import numpy as np
import pandas as pd

from contextual_bandit import ContextualBandit, ContextualBanditGreedy,  \
     ContextualBanditUCB, ContextualBanditTSUCB 
from contextual_bandit_ids import ContextualBanditIDS


@click.command()
@click.option("--seed", type=int, default=1)
@click.option("--d", type=int, default=10)
@click.option("--k", type=int, default=20)
@click.option("--t", default=5000, type=int)
@click.option("--noise_std", default=1, type=float)
@click.option("--prior_var_magnitude", default=1, type=float)
@click.option("--max_iters", default=2000, type=int)
@click.option("--alg", default='ts', type=str)
@click.option("--output", type=str, default='output.csv')
# def main(seed, d, k):
         # prior_var_magnitude):
def main(seed, d, k, t, noise_std, prior_var_magnitude, max_iters,
         alg, output):
    T = t
    def base_result():
        num_params_naive = d*k
        return dict(
            seed=seed,
            d=d,
            k=k,
            T=T,
            alg=alg,
            max_iters=max_iters,
            prior_var_magnitude=prior_var_magnitude,
            num_params_naive=num_params_naive,
            noise_std=noise_std,
        )

    filename = output
    prior_var_magnitude=prior_var_magnitude/d

    np.random.seed(seed)
    # G, gamma, true_betas = generate_params(d, d, k)
    prior_mu = np.zeros(d)
    true_betas = np.array([np.random.multivariate_normal(prior_mu,
        np.eye(d)*prior_var_magnitude) for _ in range(k)])

    groups = None
    contexts = ContextualBandit.generate_contexts(d, T)


    bs = []
    results = []

    r = base_result()
    r['alg'] = alg

    freq = False
    if alg.startswith('freq_'):
        freq = True
        alg = alg[5:]
    if alg.startswith('tsucb'):
        b = ContextualBanditTSUCB(contexts, true_betas, prior_mu, prior_var_magnitude,
                                noise_std, T, groups)
        if '_' in alg:
            num_samples = int(alg.split('_')[-1])
            b.num_samples = num_samples
    elif alg.startswith('idsprecise'):
        b = ContextualBanditIDS(contexts, true_betas, prior_mu, prior_var_magnitude,
                                noise_std, T, groups)
        if '_' in alg:
            num_samples = int(alg.split('_')[-1])
            b.num_samples = num_samples
    elif alg == 'ts':
        b = ContextualBandit(contexts, true_betas, prior_mu, prior_var_magnitude,
                             noise_std, T, groups)
    elif alg.startswith('ucb'):
        b = ContextualBanditUCB(contexts, true_betas, prior_mu, prior_var_magnitude,
                            noise_std, T, groups)
        if '_' in alg:
            ucb_alpha = float(alg.split('_')[-1])
            print('setting UCB alpha to', ucb_alpha)
            b.set_ucb_alpha(ucb_alpha)
    elif alg == 'greedy':
        b = ContextualBanditGreedy(contexts, true_betas, prior_mu,
                                   prior_var_magnitude, noise_std, T, groups)
    else:
        print(alg, 'not found!!')
    b.set_freq(freq)

    b.run_contextual()

    # attrs = []
    # sum_attrs = ['instant_regret', 'expected_reward']
    # grouped_attrs = ['expected_reward', 'instant_regret', 'optimal_reward', 'prior_expected_reward', 'greedy_reward']
    # df = b.get_grouped_results(r, attrs, sum_attrs, grouped_attrs)

    sum_attrs = ['instant_regret', 'expected_reward', 'prior_expected_reward', 'greedy_reward']
    attrs = []
    df = b.get_results(r, attrs, sum_attrs)
    # results.append(r)

    # df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print('output:', filename)


if __name__ == '__main__':
    main()
