#!/bin/bash -e

PYTHONPATH=~/Projets/bobthebuilder/attentive_planner/:~/Projets/bobthebuilder/attentive_planner/core/:${PYTHONPATH} python -u train_da2qgen_all.py
