#!/usr/bin/python
import platform
import sys
import os
plat = platform.platform()
if plat == 'macOS-10.16-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/code2/simulation_nonstationary_real/submission_scripts")
    sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/code2/simulation_nonstationary_real/submission_scripts")
elif plat == 'Linux-3.10.0-957.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real/submission_scripts")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real/submission_scripts")
elif plat == 'Linux-3.10.0-1160.45.1.el7.x86_64-x86_64-with-glibc2.17': # greatlakes
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real/submission_scripts")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real/submission_scripts")
import subprocess, time, re
import numpy as np

#%% create slurm scripts
os.system("bash sim_nonstationary_create_submissions.sh")
basis="rbf"
gammas = [0.95] #0.8,, 0.9, 0.95
gamma_names = [re.sub("\.", "", str(gamma)) for gamma in gammas]
Ns = [100]#25, , 100
# kappas = np.arange(10, 41, step=5)
kappas = np.arange(10, 41, step=5)
kappas = kappas[::-1]
# kappas = [10]
n_random_seeds = 4
njobs = 0
n_jobs_running = 0
cmd_job = ["squeue", "-u", "mengbing"]
error_handling1 = """exitStatus=1; while [[ "$exitStatus" == "1" ]]; do sleep 10; """
error_handling2 = """exitStatus=$?; done"""
for N in Ns:
    for i in range(len(gammas)):
        gamma = gammas[i]
        gamma_name = gamma_names[i]
        for kappa in kappas: #[25 + 5 * i for i in range(11)] [25,30,35,40,45,50]
            for seed in range(1, n_random_seeds+1):
                if ((kappa >= 35)):
                    continue

                job_name = '01_sim_1d_gamma' + gamma_name +\
                            '_kappa' + str(kappa) + '_N' + str(N) + '_' + str(seed) + '_run.slurm'
                cmd = error_handling1 + "sbatch " + job_name + "; " + error_handling2

                print("Submitting Job with command: %s" % cmd)
                status = subprocess.check_output(cmd, shell=True)
                time.sleep(10)
                jobnum = [int(s) for s in status.split() if s.isdigit()][0]
                print("Job number is %s" % jobnum)


                # check the number of running jobs
                # cmd_job = ["squeue", "-u", "mengbing"]
                job_status = subprocess.check_output(cmd_job)
                time.sleep(10)
                # n_jobs_running = len(job_status.split(b'\n')) - 2

                jobs_running = job_status.split(b'\n')[1:-1]
                n_jobs_running = len(jobs_running)
                # find array jobs
                array_jobs = [len(job_running.split(b'[')) for job_running in jobs_running]
                n_array_jobs = sum(array_job == 2 for array_job in array_jobs)

                while n_array_jobs > 40 or n_jobs_running > 1000:
                    time.sleep(20)
                    job_status = subprocess.check_output(cmd_job)
                    # n_jobs_running = len(job_status.split(b'\n')) - 2
                    jobs_running = job_status.split(b'\n')[1:-1]
                    n_jobs_running = len(jobs_running)
                    array_jobs = [len(job_running.split(b'[')) for job_running in jobs_running]
                    n_array_jobs = sum(array_job == 2 for array_job in array_jobs)

                njobs += 1

    print("\nCurrent status:\n")
    #show the current status
    os.system("squeue -u mengbing")



# os.system("python3 ../summary_nonstationary_rbf_1d.py")

# job_status = b'  JOBID PARTITION     NAME     USER  ACCOUNT ST       TIME  NODES NODELIST(REASON)\n 19399394_[157-500]  standard poly_tra mengbing zhenkewu PD       0:00      1 (None)\n19399394_101  standard poly_tra mengbing zhenkewu PD       0:00      1 (None)\n'
# jobs_running = job_status.split(b'\n')[1:-1]
# n_jobs_running = len(jobs_running)
# array_jobs = [len(job_running.split(b'[')) for job_running in jobs_running]
# n_array_jobs = sum(array_job == 2 for array_job in array_jobs)
#
# test1 = b'19399394_[157-500]  standard poly_tra mengbing zhenkewu PD       0:00      1 (None)\n'
# test2 = b'19399394_101  standard poly_tra mengbing zhenkewu PD       0:00      1 (None)\n'
# len(test1.split(b'['))
# len(test2.split(b'['))

# success = False
# while not success:
#     print(success)
#     time.sleep(1)
#     try:
#         status = subprocess.check_output(cmd)
#         print(status)
#         jobnum = [int(s) for s in status.split() if s.isdigit()][0]
#         if type(jobnum) == int or type(jobnum) == float:
#             success = True
#             print("Job number is %s" % jobnum)
#             # check the number of running jobs
#             job_status = subprocess.check_output(cmd_job)
#         else:
#             continue
#
#         # jobnum = [int(s) for s in status.split() if s.isdigit()][0]
#
#     except subprocess.CalledProcessError as e:
#         print(e.output)
#         raise RuntimeError(
#             "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))