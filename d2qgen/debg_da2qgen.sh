#!/bin/bash -e
export PYTHONPATH=/u/gulcehrc/Experiments/codes/python/ResearchMRC/cglr:${PYTHONPATH} 

THEANO_FLAGS="floatX=float32,device=cpu,force_device=True,optimizer=fast_compile" python -m pdb train_da2qgen_all.py
