#!/bin/bash
cd ~/cumsumrl/simulation_1d
# specify the numpy random seeds to generate data. Note that this also specifies the number of replications to run.
np_random_seeds=($(seq 1 1 100))
# specify the kappa's to run CUMSUM-RL tests on [T-kappa, T]
kappas=($(seq 25 5 75))
# specify the scenarios to run. Each element in the list contains the setting of transition,reward
settings=("homo,pwconst2" "homo,smooth" "pwconst2,homo" "smooth,homo")
# specify a list of discount factors
gammas=(0.9 0.95)
# specify the sample sizes
Ns=(25 100)
# specify the random seeds for the RBFSampler
RBFSampler_random_states=($(seq 1 1 4))
# specify the number of threads to run parallel jobs
numthreads=5


# 1. run the simulation script
echo "1. Simulate data and test for nonstationarity"
for N in "${Ns[@]}"; do
	echo "N = ${N}"
	for setting in "${settings[@]}"; do
		trans_setting=$(echo ${setting} | cut -f1 -d,)
		reward_setting=$(echo ${setting} | cut -f2 -d,)
		echo $'\n'"transition function: ${trans_setting}"
		echo "reward function: ${reward_setting}"
		for g in ${!gammas[*]}; do
			gamma=${gammas[$g]}
			echo "gamma = ${gamma}"
			for s in ${!np_random_seeds[*]}; do
				seed=${np_random_seeds[$g]}
				echo "seed = ${seed}"
				for k in ${!kappas[@]}; do
					kappa=${kappas[$k]}
					echo "kappa = ${kappa}"
					for r in ${!RBFSampler_random_states[@]}; do
						RBFSampler_random_state=${RBFSampler_random_states[$r]}
						echo "RBFSampler_random_state = ${RBFSampler_random_state}"

						python 01_sim_1d_run.py ${seed} ${kappa} ${numthreads} ${gamma} ${trans_setting} ${reward_setting} ${N} ${RBFSampler_random_state}
					done 
				done
			done
		done
	done 
done


echo "2. combine p-values from multiple random RBFSampler seeds"
python 02_combine_p_values.py 

echo "plot the rejection probabilities"
python p01_plot_combine_p_values.py


echo "3. estimate change point using isotonic regression and sequential method"
python 03_sim_1d_changept_detection_isoreg.py

echo "plot the distribution of the detected change points"
python p02_plot_changept_dist.py


# 4. evaluate policies
echo "4. Estimate optimal policies and values"
type_ests=("proposed" "overall" "oracle" "random" "kernel02" "kernel04" "kernel08" "kernel16")
for N in "${Ns[@]}"; do
	echo "N = ${N}"
	for setting in "${settings[@]}"; do
		trans_setting=$(echo ${setting} | cut -f1 -d,)
		reward_setting=$(echo ${setting} | cut -f2 -d,)
		echo $'\n'"transition function: ${trans_setting}"
		echo "reward function: ${reward_setting}"
		for g in ${!gammas[*]}; do
			gamma=${gammas[$g]}
			echo "gamma = ${gamma}"
			for s in ${!np_random_seeds[*]}; do
				seed=${np_random_seeds[$g]}
				echo "seed = ${seed}"
				for type_est in "${type_ests[@]}"; do
					echo "type_est = ${type_est}"

					python 04_sim_1d_changept_optvalue_run.py ${seed} ${trans_setting} ${reward_setting} ${gamma} ${N} ${type_est}
				done
			done
		done
	done 
done

echo "plot the values of different policies"
python p03_plot_changept_value.py
