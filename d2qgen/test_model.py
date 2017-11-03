'''
    Build a neural machine translation model with soft attention
'''


from __future__ import print_function

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data.squad import SQUADIterator
from data import DataMode
from nunits import NSigmoid, NTanh
from model_utils import zipp, unzip, itemlist, load_params, init_tparams, \
        surround, print_params, ensure_dir_exists


from optimizers import adam, adadelta, rmsprop, sgd
from model import *
import pprint as pp


profile = False


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=False):
    probs = []
    n_done = 0

    for batch in iterator:
        doc, doc_mask, q, q_mask, ans, ans_locs, ans_mask = batch
        n_done += len(doc)
        doc, q, ans, ans_locs = prepare_data(doc, doc_mask, q, q_mask, ans, ans_locs,
                                             ans_mask,
                                             max_len=options['maxlen'])

        if doc is not None and q is not None:
            if options['use_doc_emb_ans']:
                pprobs = f_log_probs(doc, doc_mask, q,
                                     q_mask, ans_locs,
                                     ans_mask)
            else:
                pprobs = f_log_probs(doc, doc_mask, q,
                                     q_mask, ans, ans_mask)

            for pp in pprobs:
                probs.append(pp)

            if numpy.isnan(numpy.mean(probs)):
                ipdb.set_trace()

            if verbose:
                print('%d samples computed' % (n_done))

    return numpy.array(probs)


def evaluate(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_doc=100000,  # source vocabulary size
          n_words_q=100000,  # target vocabulary size
          n_words_ans=10000,
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          use_pointer_softmax=False,
          saveto='model.npz',
          validFreq=1000,
          temp_switch=1.0,
          assign_priority_point=False,
          use_batch_norm=False, # Whether to use the batch norm or not.
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          use_att_embeds=True,
          debug=False,
          dict_dir="dicts/",
          use_char_ans=False,
          use_bpe=False,
          use_doc_emb_ans=False,
          datasets=None,
          val_size=-1,
          use_dropout=False,
          use_constrained_training=False,
          constraints=None,
          ans_cost_lambda=1e-3,
          do_planning=False,
          condition_on_context=False,
          shift_before=False):

    # Model options
    model_options = locals().copy()
    pp.pprint(model_options)

    # reload options
    reload_ = True
    if reload_ and os.path.exists(saveto):
        print("Reloading options...")
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)
        print("Loaded successfully.")

    print('Loading data')
    test = SQUADIterator(path=datasets,
                          nwords_doc=n_words_doc,
                          nwords_q=n_words_q,
                          use_pointers=use_pointer_softmax,
                          nwords_ans=n_words_ans,
                          use_doc_emb_ans=use_doc_emb_ans,
                          assign_priority_point=assign_priority_point,
                          maxlen=maxlen,
                          batch_size=batch_size)

    n_words_doc, n_words_q, n_words_ans = test.nwords_doc, test.nwords_q, \
                                          test.nwords_ans

    model_options['n_words_doc'] = n_words_doc + 1
    model_options['n_words_q'] = n_words_q + 1
    model_options['n_words_ans'] = n_words_ans + 1
    model_options['n_words'] = n_words_q + 1

    print("number words in the answer, ", model_options['n_words_ans'])


    if model_options['use_pointer_softmax']:
        model_options['maxlen'] = test.maxlen

    ensure_dir_exists(dict_dir)

    if use_pointer_softmax:
        common_words_dict = test.common_words_dict
        common_words_dict_r = {v:k for k,v in common_words_dict.iteritems()}
    else:
        common_words_dict = None
        common_words_dict_r = None

    print('Building model')
    model = Model(prefix="da2qmodel", options=model_options)

    params = model.init_params()

    # reload parameters
    if reload_ and os.path.exists(saveto):
        print("Reloading params...")
        params = load_params(saveto, params)
        print("Loaded params successfully.")

    tparams = init_tparams(params)
    print_params(tparams)

    trng, use_noise, \
        doc, doc_mask, q, q_mask, \
        ans, ans_mask, \
        opt_ret, \
        cost, \
        updates = \
        model.build_model(tparams, None)
    inps = [doc, doc_mask, q, q_mask, ans, ans_mask]

    # before any regularizer
    print('Building f_log_probs...',)

    # Compile the Theano function to compute the log-probs.
    f_log_probs = theano.function(inps, cost, updates=updates, profile=profile)


    print("Getting the error:")
    test.reset()
    use_noise.set_value(0.)
    test_errs = pred_probs(f_log_probs,
                                        prepare_data,
                                        model_options,
                                        test)

    test_err = test_errs.mean()
    print('Error: ', test_err)

    return test_err

def main(job_id, params):
    data_path = "/data/lisatmp4/gulcehrc/data/squad/"
    datasets = os.path.join(data_path, 'dev-v1.1.json')

    debug = False
    validerr = evaluate(saveto=params['model'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words_q=params['n_words_q'],
                     n_words_doc=params['n_words_doc'],
                     n_words_ans=params['n_words_ans'],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     use_pointer_softmax=True,
                     debug=debug,
                     use_batch_norm=True,
                     patience=1000,
                     datasets=datasets,
                     batch_size=16,#32
                     temp_switch=params['temp_switch'],
                     valid_batch_size=16,
                     validFreq=1200,
                     use_doc_emb_ans=params['use_doc_emb_ans'],
                     dispFreq=20,
                     maxlen=660,
                     assign_priority_point=True,
                     saveFreq=3000,
                     sampleFreq=600,
                     use_dropout=params['use-dropout'][0],
                     do_planning=False,#True,
                     val_size=2000,
                     #condition_on_context=True,
                     #shift_before=False,
                     )
    return validerr


if __name__ == '__main__':
    m_path = "/data/lisatmp4/dutilfra/d2qgen/baseline5/"
    #m_path = "/data/lisatmp4/dutilfra/d2qgen/norepeat_10_4/"
    #m_path = "/data/lisatmp4/dutilfra/d2qgen/norepeat_10_context_3/"
    #m_path = "/data/lisatmp4/dutilfra/d2qgen/norepeat_10_context_add/"

    main(0, {
        'model': [m_path + \
                  'model_pointer_softmax_model_temp_switch=1_uadam_qgenS.npz_best.npz'],
        'dim_word': [600],
        'dim': [800],
        'n_words_doc': 64000,
        'n_words_ans': 30000,
        'n_words_q': 2000,
        'optimizer': ['adadelta'],
        'decay-c': [5e-5],
        'clip-c': [1.2],
        'use-dropout': [True],
        'learning-rate': [0.0002],
        'temp_switch': 1.0,
        'use_doc_emb_ans': True,}
        )
