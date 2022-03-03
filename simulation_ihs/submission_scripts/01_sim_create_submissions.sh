#!/bin/bash
cd ~/cumsumrl/simulation_ihs/submission_scripts

gammas=(0.9 0.95) 
gamma_names=(09 095) 
random_seeds=($(seq 1 1 4))
N=100

# ${!array[@]} is the list of all the indexes set in the array
for i in ${!gammas[@]}; do
    gamma=${gammas[$i]}

    for kappa in {10..40..5}; do
        time=02:00:00
        for s in ${!random_seeds[@]}; do
            seed=${random_seeds[$s]}
    echo "#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=ihs_gamma${gamma_names[$i]}_kappa${kappa}_${seed}
#SBATCH --time=${time}
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=4g
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100
#SBATCH -o ./reports/%x_%A_%a.out

module load python
cd ~/cumsumrl/simulation_ihs

python3 01_sim_ihs_run.py \$SLURM_ARRAY_TASK_ID ${kappa} ${gamma} ${N} ${seed}" > 01_sim_ihs_gamma${gamma_names[$i]}_kappa${kappa}_N${N}_${seed}_run.slurm

        done
    done
done
