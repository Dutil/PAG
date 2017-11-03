#!/bin/bash
# -*- coding: utf-8 -*-

#sbatch --gres=gpu -C"gpu12gb" -t 3-0 --mail-type=ALL --mail-user=frdutil@gmail.com run_exp_planning_Gumbel.sh

echo $HOSTNAME
#echo "Copying the dataset..."
#if [ ! -d /Tmp/dutilfra/nmt/data/wmt15/ ]; then
#    mkdir -p /Tmp/dutilfra/nmt/data/wmt15/
#    cp -a /data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/ /Tmp/dutilfra/nmt/data/wmt15/
#    echo "Done."
#else
#    echo "Already there"
#fi
#echo "Done."

echo "Starting the training"
export PYTHONPATH="$PYTHONPATH:/u/dutilfra/Projets/PAG/"
python -u train_graph_adam_baseline.py graph_baseline.txt
