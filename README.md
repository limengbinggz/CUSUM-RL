# Reinforcement Learning in Possibly Nonstationary Environments (CUSUM-RL)

This repository contains the implementation for the paper "Reinforcement Learning in Possibly Nonstationary Environments" in Python (and R for plotting).


## Summary of the paper

We consider reinforcement learning (RL) methods in offline nonstationary environments. Many existing RL algorithms in the literature rely on the stationarity assumption that requires the system transition and the reward function to be constant over time. However, the stationarity assumption is restrictive in practice and is likely to be violated in a number of applications, including traffic signal control, robotics and mobile health. In this paper, we develop a consistent procedure to test the nonstationarity of the optimal policy based on pre-collected historical data, without additional online data collection. Based on the proposed test, we further develop a sequential change point detection method that can be naturally coupled with existing state-of-the-art RL methods for policy optimisation in nonstationary environments. The usefulness of our method is illustrated by theoretical results, simulation studies, and a real data example from the [2018 Intern Health Study (IHS)](https://www.srijan-sen-lab.com/intern-health-study).


## File Overview

- Folder `functions/`: This folder contains all utility Python functions used in numerical experiments including simulation and data analysis
    - `compute_test_statistics.py` implements the proposed CUMSUM-RL test of nonstationarity.
    - `evaluation.py` implements the evaluation procedure. Specifically, it contains functions for estimating the optimal policy and estimating the value of the policy using fitted-Q evaluation.
    - `simulate_data_1d.py` generates data in 1-dimensional simulation. It contains functions to simulate data in 4 scenarios of different transition and reward functions.
    - `simulate_data_real.py` generated data in the IHS study simulation.

- Folder `simulation_1d/`: This folder contains the platform that realizes the 1-dimensional simulation in the paper. Numbers prefixing the names of the .py files indicate the order to realize the simulation scenarios. Files starting with a p in their names contain codes to generate plots in the paper. 
    - `01_sim_1d_run.py` simulates 1-dimensional data and test for nonstationarity on a specified time interval. Usage:
    ```console
    python 01_sim_1d_run.py {seed} {kappa} {num_thread} {gamma} {trans_setting} {reward_setting} {N} {RBFSampler_random_seed}
    ```
    See the annotation in the script for the meanings of arguments. Example:
    ```console
    python 01_sim_1d_run.py 2 30 5 0.9 homo smooth 25 1
    ```
    - `02_combine_p_values.py` aggregates p-value results from multiple random seeds in RBFSampler, with a specified quantile.
    - `03_sim_1d_changept_detection_isoreg.py` estimates change points using isotonic regression.
    - `04_sim_1d_changept_optvalue_run.py` estimates the optimal policies and values using different methods.
    - `p01_plot_combine_p_values.py` creates Figure 3 in paper of rejection probabilities. 
    - `p02_plot_changept_dist.py` creates Figure 4 in paper of distribution of the estimated change points.
    - `p03_plot_changept_value.py` creates Figure 5 in paper of optimal values of different estimated policies.
    - To run the 1-dimensional simulation in sequence, 
    ```sh
    bash run_sim_1d.sh
    ```
    - To run the 1-dimensional testing on a cluster and submit the simulation jobs using `slurm` by sample size, kappa, gamma, and data settings,
    ```console
    python submission_scripts/01_sim_submit.py
    ```
    Next, run `02_combine_p_values.py` and `03_sim_1d_changept_detection_isoreg.py`.
    Finally, split jobs again for evaluation by submitting
    ```console
    python submission_scripts/04_sim_submit.py
    ```
    - Folder `output` contains raw results and corresponding figures of the simulation in the paper.

- Folder `simulation_ihs/`: This folder contains the platform that realizes the IHS simulation in the paper. Numbers prefixing the names of the .py files indicate the order to realize the simulation scenarios. Files starting with a p in their names contain codes to generate plots in the paper. 
    - `01_sim_ihs_run.py` simulates IHS data and test for nonstationarity on a specified time interval. Usage:
    ```console
    python 01_sim_ihs_run.py {seed} {kappa} {gamma} {N} {RBFSampler_random_seed}
    ```
    See the annotation in the script for the meanings of arguments. Example:
    
    ```console
    python 01_sim_ihs_run.py 2 25 0.9 100 1
    ```
    - `02_combine_p_values.py` aggregates p-value results from multiple random seeds in RBFSampler, with a specified quantile.
    - `03_sim_ihs_changept_detection_isoreg.py` estimates change points using isotonic regression.
    - `04_sim_ihs_changept_optvalue_run.py` estimates the optimal policies and values using different methods.
    - `p01_plot_combine_p_values.py` creates Figure 6(a) in paper of rejection probabilities. 
    - `p02_plot_changept_dist.py` creates Figure 6(b) in paper of distribution of the estimated change points.
    - `p03_plot_changept_value.py` creates Table 2 in paper of optimal values of different estimated policies.
    - To run the IHS simulation in sequence, 
    ```sh
    bash run_sim_ihs.sh
    ```

    - To run the IHS testing on a cluster and submit the simulation jobs using `slurm` by sample size, kappa, gamma, and data settings,
    $ python submission_scripts/01_sim_submit.py
    Next, run `02_combine_p_values.py` and `03_sim_ihs_changept_detection_isoreg.py`.
    Finally, split jobs again for evaluation by submitting
    ```console
    python submission_scripts/04_sim_submit.py
    ```
    - Folder `output` contains raw results and corresponding figures of the simulation in the paper.

