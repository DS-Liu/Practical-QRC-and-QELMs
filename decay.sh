#!/bin/sh
#SBATCH -J decay -o job-%j-%x.out -e job-%j-%x.out
#SBATCH -p CPU-192C768GB -n 17 --qos=qos_cpu_192c768gb
export OMP_NUM_THREADS=1
. /etc/profile.d/modules.sh

echo Time is `date`, Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST, allocated $SLURM_JOB_CPUS_PER_NODE cpu core, $OMP_NUM_THREADS threads for each task

mpiexec -n $SLURM_NTASKS python -m mpi4py.futures ./decay.py --qrc_evolve
mpiexec -n $SLURM_NTASKS python -m mpi4py.futures ./decay.py --qelm_evolve