  #!/bin/bash
#$ -cwd
#$ -N medsam_debug
#$ -o /cbica/home/slavkovk/project_medsam_testing/Scripts/cluster_output/$JOB_NAME_$JOB_ID.output
#$ -j y

#### REMEMBER TO EDIT 1) "-N" and 2) python script file name 

#$ -M kslav@sas.upenn.edu
#$ -m b #### send mail at the beginning of the job
#$ -m e #### send mail at the end of the job
#$ -m a #### send mail in case the job is aborted
#$ -l h_vmem=30G
#$ -l h_rt=02:00:00

source activate medsam
python /cbica/home/slavkovk/project_medsam_testing/Training/utils/convert_3Dto2D.py
deactivate 
