#!/bin/bash
cd ~/cumsumrl/simulation_ihs/submission_scripts

gammas=(0.9 0.95)
gamma_names=(09 095)
type_ests=("overall" "proposed" "oracle" "random")
N=100

for i in ${!gammas[@]}; do
    gamma=${gammas[$i]}
    for type_est in "${type_ests[@]}"; do

	echo "#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=ihs_gamma${gamma_names[$i]}_N${N}_${type_est}
#SBATCH --time=5:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=4g
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100
#SBATCH -o ./reports/%x_%A_%a.out 

module load python
cd ~/cumsumrl/simulation_ihs

python3 04_sim_ihs_changept_optvalue_run.py \$SLURM_ARRAY_TASK_ID ${gamma} ${N} ${type_est}" > 04_sim_ihs_gamma${gamma_names[$i]}_N${N}_${type_est}_run.slurm
#	 sbatch sim_nonstationary_gamma${gamma_names[$i]}_N${N}_${type_est}_run.slurm


	done
done
