#!/bin/bash

./make-toy-data --dims 3 --producer gmm 200 ./out.npz
./analyze --analysis projection --output ./plot.png \
    --color-by ground_truth --shape-by predictions \
    --keep-percentage 0.2 ./out.npz
./analyze --analysis nmi -i ./out.npz
./analyze --analysis cluster_purity -k 3 -i ./out.npz
