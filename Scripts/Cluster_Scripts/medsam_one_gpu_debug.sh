  #!/bin/bash
#$ -cwd
#$ -N medsam_train_val
#$ -o /cbica/home/slavkovk/project_medsam_testing/Scripts/cluster_output/$JOB_NAME_$JOB_ID.output
#$ -j y

#### REMEMBER TO EDIT 1) "-N" and 2) python script file name 

#$ -M kslav@sas.upenn.edu
#$ -m b #### send mail at the beginning of the job
#$ -m e #### send mail at the end of the job
#$ -m a #### send mail in case the job is aborted
#$ -l h_vmem=50G
#$ -l A40
#$ -l h_rt=00:10:00

source activate medsam
nvidia-smi
module load cuda/11.8
python /cbica/home/slavkovk/project_medsam_testing/main.py --config /cbica/home/slavkovk/project_medsam_testing/configs/config_default.json
module unload cuda/11.8
source deactivate 
