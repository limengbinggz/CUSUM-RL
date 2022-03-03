#!/bin/bash
cd ~/cumsumrl/simulation_1d/submission_scripts

gammas=(0.9 0.95)
gamma_names=(09 095)
random_seeds=($(seq 1 1 4))
Ns=(25 100)
numthreads=5

for setting in "homo","pwconst2" "homo","smooth" "pwconst2","homo" "smooth","homo"; do
	IFS=',' read trans_setting reward_setting <<< "${setting}"
    echo "${trans_setting}" and "${reward_setting}"

	# ${!array[@]} is the list of all the indexes set in the array
	for i in ${!gammas[@]}; do
		gamma=${gammas[$i]}

		for kappa in {25,30,35,40,45,50,55,60,65,70,75}; do
			for N in "${Ns[@]}"; do
		        for s in ${!random_seeds[@]}; do
				            seed=${random_seeds[$s]}

	echo "#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=1d_trans${trans_setting}_reward${reward_setting}_gamma${gamma_names[$i]}_kappa${kappa}_N${N}_${seed}
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=2g
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100
#SBATCH -o ./reports/%x_%A_%a.out 

module load python
cd ~/cumsumrl/simulation_1d

python3 01_sim_1d_run.py \$SLURM_ARRAY_TASK_ID ${kappa} ${numthreads} ${gamma} ${trans_setting} ${reward_setting} ${N} ${seed}
" > 01_sim_1d_trans${trans_setting}_reward${reward_setting}_gamma${gamma_names[$i]}_kappa${kappa}_N${N}_${seed}_run.slurm

				done
			done
		done
	done
done
