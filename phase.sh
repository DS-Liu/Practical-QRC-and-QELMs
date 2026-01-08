#!/bin/sh
#SBATCH -J phase -o job-%j-%x.out -e job-%j-%x.out
#SBATCH -p CPU-192C768GB -N 6 -n 1001 -c 1 --qos=qos_cpu_192c768gb
export OMP_NUM_THREADS=1 # This is very important, otherwise the routine will run extremely slow.
. /etc/profile.d/modules.sh

echo Time is `date`, Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST, allocated $SLURM_JOB_CPUS_PER_NODE cpu core, $OMP_NUM_THREADS threads for each task

mpiexec -n $SLURM_NTASKS python -m mpi4py.futures phase.py