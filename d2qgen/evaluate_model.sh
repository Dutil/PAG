#!/bin/bash -e

echo "Evaluating the baseline..."
cp ~/tmplisa4/d2qgen/baseline5/model_pointer_softmax_model_temp_switch=1_uadam_qgenS.npz.pkl ~/tmplisa4/d2qgen/baseline5/model_pointer_softmax_model_temp_switch=1_uadam_qgenS.npz_best.npz.pkl
THEANO_FLAGS='floatX=float32, device=gpu, optimizer=fast_run, optimizer_including=alloc_empty_to_zeros, lib.cnmem=1' python -u test_model.py
