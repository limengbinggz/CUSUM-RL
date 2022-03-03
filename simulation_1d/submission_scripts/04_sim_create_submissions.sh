#!/bin/bash
cd ~/cumsumrl/simulation_1d/submission_scripts

basis="rbf"
gammas=(0.9 0.95)
gamma_names=(09 095)
Ns=(25 100)
type_ests=("proposed" "overall" "oracle" "random" "kernel02" "kernel04" "kernel16" "kernel08")

for setting in "homo","pwconst2" "homo","smooth" "pwconst2","homo" "smooth","homo"; do
	IFS=',' read trans_setting reward_setting <<< "${setting}"
    echo "${trans_setting}" and "${reward_setting}"

	# ${!array[@]} is the list of all the indexes set in the array
	for i in ${!gammas[@]}; do
		gamma=${gammas[$i]}
		for N in "${Ns[@]}"; do

      for type_est in "${type_ests[@]}"; do


	echo "#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=1d_trans${trans_setting}_reward${reward_setting}_gamma${gamma_names[$i]}_N${N}_${type_est}
#SBATCH --time=3:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=5g
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100
#SBATCH -o ./reports/%x_%A_%a.out 

module load python
cd ~/cumsumrl/simulation_1d

python3 04_sim_1d_changept_optvalue_run.py \$SLURM_ARRAY_TASK_ID ${trans_setting} ${reward_setting} ${gamma} ${N} ${type_est}" > 04_sim_1d_trans${trans_setting}_reward${reward_setting}_gamma${gamma_names[$i]}_N${N}_${type_est}_run.slurm
	# sbatch sim_nonstationary_trans${trans_setting}_reward${reward_setting}_gamma${gamma_names[$i]}_run.slurm

      done
		done
	done
done
