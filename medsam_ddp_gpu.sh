#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --job-name=medsam_ddp
#SBATCH --gpus=a40:2
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-gpu=4
#SBATCH --output=Scripts/cluster_output/mgpus_%x-%j.out
#SBATCH --error=Scripts/cluster_output/mgpus_%x-%j.err
#SBATCH --time=00:10:00

# modified from /MedSAM/train_multi_gpus.sh for one node, multiple GPUs
# doesn't successfully launch in this current state

set -x -e
# load the modules and environment
module load cuda/11.8
source activate medsam

# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_JOB_CPUS_ON_NODE" = $SLURM_JOB_CPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR

# Training setup
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

## Master node setup
MAIN_HOST=`hostname -s`
export MASTER_ADDR=$MAIN_HOST

# Get a free port using python
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

export NNODES=$SLURM_NNODES
#NODE_RANK=$SLURM_PROCID ## do i need this?
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs

echo "nnodes: ${NNODES}"

## Vector's cluster doesn't support infinite bandwidth
## but gloo backend would automatically use inifinite bandwidth if not disable
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

export NCCL_DEBUG=INFO

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
echo SLURM_NTASKS=$SLURM_NTASKS


srun python /cbica/home/slavkovk/project_medsam_testing/main_ddp.py \
        --config /cbica/home/slavkovk/project_medsam_testing/configs/config_default.json \
        --world_size ${WORLD_SIZE} \
        --node_rank 0 \
        --init_method tcp://${MASTER_ADDR}:${MASTER_PORT} 


echo "END TIME: $(date)"
source deactivate medsam
