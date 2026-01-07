#!/bin/sh
#SBATCH -J otoc -o job-%j-%x.out -e job-%j-%x.out
#SBATCH -p test -N 1 -n 10 --qos=qos_test
export OMP_NUM_THREADS=1 # This is very important, otherwise the routine will run extremely slow.
. /etc/profile.d/modules.sh

echo Time is `date`, Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST, allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

python ./otoc.py