#!/bin/bash
export PYTHONPATH=../core:$PYTHONPATH
export PYTHONPATH=../charnmt:$PYTHONPATH
export PYTHONPATH=../:$PYTHONPATH

THEANO_FLAGS="floatX=float32,device=gpu,force_device=True,lib.cnmem=0.95" python train_graph_adam_planning.py graph_plan_4.txt
