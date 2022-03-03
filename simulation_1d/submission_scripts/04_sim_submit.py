#!/usr/bin/python
import platform, sys, os, re, subprocess, time
import numpy as np
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d/submission_scripts")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d/submission_scripts")

#%% create slurm scripts
settings = [["homo","pwconst2"], ["homo","smooth"], ["pwconst2","homo"], ["smooth","homo"]]
os.system("bash 04_sim_create_submissions.sh")
gammas = [0.9,0.95]
type_ests = ['random', 'overall', 'oracle', 'kernel02', 'kernel04', 'kernel08', 'kernel16']
Ns = [25,100]
gamma_names = [re.sub("\.", "", str(gamma)) for gamma in gammas]

njobs = 0
n_jobs_running = 0
cmd_job = ["squeue", "-u", "mengbing"]# !!change to your account name
for N in Ns:
    for setting in settings:
        trans_setting = setting[0]
        reward_setting = setting[1]
        for i in range(len(gammas)):
            gamma = gammas[i]
            gamma_name = gamma_names[i]
            for type_est in type_ests:

                job_name = '04_sim_1d_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + gamma_name +\
                            '_N' + str(N) + '_' + type_est + '_run.slurm'
                cmd = ["sbatch", job_name]
                print("Submitting Job with command: %s" % cmd)
                status = subprocess.check_output(cmd)
                jobnum = [int(s) for s in status.split() if s.isdigit()][0]
                print("Job number is %s" % jobnum)

                # check the number of running jobs
                job_status = subprocess.check_output(cmd_job)
                time.sleep(10)

                jobs_running = job_status.split(b'\n')[1:-1]
                n_jobs_running = len(jobs_running)
                # find array jobs
                array_jobs = [len(job_running.split(b'[')) for job_running in jobs_running]
                n_array_jobs = sum(array_job == 2 for array_job in array_jobs)

                # if more than 4 array jobs or 1000 individual jobs are running simutaneously, we will wait
                # until they finish (in case the cluster does not allow for submitting too many jobs at once)
                while n_array_jobs > 5 or n_jobs_running > 1500:
                    time.sleep(60)
                    job_status = subprocess.check_output(cmd_job)
                    jobs_running = job_status.split(b'\n')[1:-1]
                    n_jobs_running = len(jobs_running)
                    array_jobs = [len(job_running.split(b'[')) for job_running in jobs_running]
                    n_array_jobs = sum(array_job == 2 for array_job in array_jobs)

                njobs += 1

    print("\nCurrent status:\n")
    #show the current status
    os.system("squeue -u mengbing")
