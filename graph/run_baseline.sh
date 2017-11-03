#!/bin/bash 
export PYTHONPATH=../core:$PYTHONPATH
export PYTHONPATH=../charnmt:$PYTHONPATH 
export PYTHONPATH=../:$PYTHONPATH 

THEANO_FLAGS="floatX=float32,device=gpu,force_device=True,lib.cnmem=0.95" python -m pdb train_graph_adam_baseline.py graph_model.txt
