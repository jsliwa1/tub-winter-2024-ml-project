#!/bin/bash

#SBATCH --partition=gpu-teaching-5h   # Specify the partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --job-name=run_python_script  # Set a job name
#SBATCH --output=job_%j.log           # Save output to a log file (job ID will be used in the filename)

# Generate a unique session name using the current timestamp
SESSION_NAME="session_$(date +%Y%m%d_%H%M%S)"

apptainer exec --nv -B /home/space/datasets:/mnt/datasets pml.sif python "$1"
