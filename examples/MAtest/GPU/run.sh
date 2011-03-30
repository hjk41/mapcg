#!/bin/sh

for n in 100 200 1000; do
	for b in 4 8 16 32 64 128; do
		for grid in 1 2 4 8 16 32 64 128; do
			for block in 1 2 4 8 16 32 64 128 256; do
				echo "n=$n b=$b g=$grid bl=$block"
				./builtin $n $b $grid $block
			done
		done
	done
done
