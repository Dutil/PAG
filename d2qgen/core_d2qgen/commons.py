import logging
import sys

import numpy

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

EPS = 1e-6
DEFAULT_SEED = 3
floatX = theano.config.floatX

global_rng = numpy.random.RandomState(DEFAULT_SEED)
global_trng = RandomStreams(DEFAULT_SEED)

sigmoid = lambda x, use_noise=0: TT.nnet.sigmoid(x)
softmax = lambda x : TT.nnet.softmax(x)
tanh = lambda x, use_noise=0: TT.tanh(x)

#Rectifier nonlinearities
rect = lambda x, use_noise=0: 0.5 * (x + abs(x + 1e-4))


leaky_rect = lambda x, leak=0.95, use_noise=0: ((1 + leak) * x + (1 - leak) * abs(x)) * 0.5
trect = lambda x, use_noise=0: Rect(Tanh(x + EPS))
trect_dg = lambda x, d, use_noise=0: Rect(Tanh(d*x))

softmax = lambda x: TT.nnet.softmax(x)
linear = lambda x: x

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Change this to change the location of parent directory where your models will be
# dumped into.
SAVE_DUMP_FOLDER="./"

