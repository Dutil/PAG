#!/bin/bash -e

#sbatch --gres=gpu --mem=6000 --mail-type=ALL --mail-user=frdutil@gmail.com run_da2qgen_plan.sh


#export PYTHONPATH=/u/gulcehrc/Experiments/codes/python/ResearchMRC/cglr:${PYTHONPATH}
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#THEANO_FLAGS="floatX=float32,device=gpu0,force_device=True,lib.cnmem=0.92" python -m pdb train_da2qgen_all.py


#export PYTHONPATH=~/Projets/bobthebuilder/attentive_planner/:~/Projets/bobthebuilder/attentive_planner/core/:${PYTHONPATH}
export PYTHONPATH="$PYTHONPATH:/workspace/bobthebuilder/attentive_planner/core:/workspace/bobthebuilder/attentive_planner/"
THEANO_FLAGS='floatX=float32, device=gpu, optimizer=fast_run, optimizer_including=alloc_empty_to_zeros, lib.cnmem=1' python -u train_da2qgen_plan.py
