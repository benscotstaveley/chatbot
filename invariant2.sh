#!/bin/bash

for seed in {0..5}
do
    for iter in {0..5}
    do
	python quick.py $seed |sed -e '1,/---sampled tokens---/d' > output_${seed}_${iter}
    done
done
