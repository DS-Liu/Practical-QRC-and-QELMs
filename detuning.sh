#!/bin/sh
#SBATCH -J detuning -o job-%j-%x.out -e job-%j-%x.out
#SBATCH -p CPU-192C768GB -n 14 --qos=qos_cpu_192c768gb
export OMP_NUM_THREADS=1

echo Time is `date`, Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST, allocated $SLURM_JOB_CPUS_PER_NODE cpu cores, $OMP_NUM_THREADS threads for each task
. /etc/profile.d/modules.sh

 mpiexec -n $SLURM_NTASKS python -m mpi4py.futures ./detuning.py --qrc_evolve
 mpiexec -n $SLURM_NTASKS python -m mpi4py.futures ./detuning.py --qelm_evolve