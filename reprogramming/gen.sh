#!/bin/bash

lamda_vals=(0.0125 0.025 0.05 0.1 0.2)
# N_vals=(64 32 16 8 4 2 1)
N_vals=(10)
srand_vals=$(seq 1 10)

for lamda in ${lamda_vals[@]}; do
	for N in ${N_vals[@]}; do
		for srand in ${srand_vals[@]}; do 
			suffix="lamda_"$lamda"_N_"$N"_srand_"$srand
			runfile="run_"$suffix".sh"
			cp run.sh $runfile
			sed -i "s/__SRAND__/$srand/g" $runfile
			sed -i "s/__N__/$N/g" $runfile
			sed -i "s/__LAMDA__/$lamda/g" $runfile
			sbatch $runfile
		done
	done
done
