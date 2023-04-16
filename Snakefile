import os

base_dir = 'results'

d = 10
k = [3, 5, 10, 20, 60]
T=10000
noise_std=[0.05, 0.1, 0.5, 1, 2]

seed=range(0, 200)
algs = ['tsucb', 'ts', 'greedy', 'freq_ucb', 'tsucb_100']


rule all:
    input:
        expand('{base_dir}/alg{alg}_d{d}_k{k}_T{T}_seed{seed}_noise_std{noise_std}.csv',
                base_dir=base_dir,
                d=d,
                k=k,
                T=T,
                seed=seed,
                noise_std=noise_std,
                alg=algs,
                ),

rule do_one_run:
    output:
        os.path.join(base_dir, 'alg{alg}_d{d}_k{k}_T{T}_seed{seed}_noise_std{noise_std}.csv')
    shell:
        'python3 run_one.py'
        ' --d={wildcards.d}'
        ' --k={wildcards.k}'
        ' --t={wildcards.T}'
        ' --seed={wildcards.seed}'
        ' --noise_std={wildcards.noise_std}'
        ' --alg={wildcards.alg}'
        ' --output={output}'


