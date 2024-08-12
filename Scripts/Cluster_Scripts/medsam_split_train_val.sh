  #!/bin/bash
#$ -cwd
#$ -N data_split
#$ -o /ifs/scratch/dk3360_gp/kps2152/job_outputs/$JOB_NAME_$JOB_ID.output
#$ -j y

#### REMEMBER TO EDIT 1) "-N" and 2) python script file name 

#$ -M kslav@sas.upenn.edu
#$ -m b #### send mail at the beginning of the job
#$ -m e #### send mail at the end of the job
#$ -m a #### send mail in case the job is aborted
#$ -l h_vmem=30G
#$ -l h_rt=00:50:00

source activate medsam
python /ifs/home/dk3360_gp/kps2152/project_medSAM_testing/Training/utils/split_Data.py
deactivate 
