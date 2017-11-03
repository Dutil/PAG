#!/bin/bash
# -*- coding: utf-8 -*-

echo $HOSTNAME
echo "Starting the training"
export PYTHONPATH="$PYTHONPATH:/u/dutilfra/Projets/PAG/charnmt:/u/dutilfra/Projets/PAG/"
python -u ../train_wmt15_adam_planning.py ./deen_norepeat.txt
