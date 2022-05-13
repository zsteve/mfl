#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --account=$ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=8192M

ml load python
ml load scipy-stack
source $HOME/langevin_env/bin/activate

N=__N__
SRAND=__SRAND__
LAMDA=__LAMDA__

python main.py --gwot --N $N --srand $SRAND --lamda_gwot $LAMDA --outfile "out_gwot_N_"$N"_srand_"$SRAND"_lamda_"$LAMDA".npy"
