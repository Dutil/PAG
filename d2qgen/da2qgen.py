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


# generate sample, either with stochastic
# sampling or beam search. Note that,
# this function iteratively calls
# f_init and f_next functions.
def gen_sample(tparams, f_init,
               f_next, doc, ans, doc_mask,
               ans_mask, options,
               trng=None, k=1,
               maxlen=30,
               stochastic=True,
               argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(doc, ans, doc_mask, ans_mask)
    next_state, ctx0, x_emb = ret[0], ret[1], ret[2]

    next_commit = numpy.ones((1, 10)).astype('float32')
    next_ctxs = numpy.zeros((1, 2 * options['dim'])).astype('float32')
    next_action_plan = numpy.ones((x_emb.shape[0], 1, 10)).astype('float32')

    next_w = (-1 * numpy.ones((1,))).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state, x_emb, ans, doc_mask, ans_mask, next_commit, next_action_plan, next_ctxs]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]
        next_commit, next_action_plan = ret[3], ret[4]
        next_ctxs = ret[5]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]

            sample.append(nw)
            sample_score += next_p[0, nw]

            if nw == 0 or nw == x_emb.shape[0]:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_commits = []
            new_hyp_actions = []
            new_hyp_ctxs = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_hyp_commits.append(copy.copy(next_commit[ti]))
                new_hyp_actions.append(copy.copy(next_action_plan[ti]))
                new_hyp_ctxs.append(copy.copy(next_ctxs[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_commits = []
            hyp_actions = []
            hyp_ctxs= []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0 or new_hyp_samples[idx][-1] == x_emb.shape[0]:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_commits.append(new_hyp_commits[idx])
                    hyp_actions.append(new_hyp_actions[idx])
                    hyp_ctxs.append(new_hyp_ctxs[idx])

            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            next_commit = numpy.array(hyp_commits)
            next_action_plan = numpy.array(hyp_actions)
            next_ctxs = numpy.array(hyp_ctxs)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


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


def train(dim_word=100,  # word vector dimensionality
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
          datasets=['', ''],
          val_size=-1,
          use_dropout=False,
          use_constrained_training=False,
          constraints=None,
          ans_cost_lambda=1e-3,
          reload_=False,
          do_planning=True,
          condition_on_context=False,
          shift_before=False):

    # Model options
    model_options = locals().copy()
    pp.pprint(model_options)

    #import ipdb
    #ipdb.set_trace()

    # reload options
    if reload_ and os.path.exists(saveto):
        print("Reloading options...")
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)
        print("Loaded successfully.")

    print('Loading data')
    train = SQUADIterator(path=datasets[0],
                          nwords_doc=n_words_doc,
                          nwords_q=n_words_q,
                          use_pointers=use_pointer_softmax,
                          nwords_ans=n_words_ans,
                          use_doc_emb_ans=use_doc_emb_ans,
                          assign_priority_point=assign_priority_point,
                          maxlen=maxlen,
                          val_size=val_size,
                          batch_size=batch_size)

    n_words_doc, n_words_q, n_words_ans = train.nwords_doc, train.nwords_q, \
            train.nwords_ans

    model_options['n_words_doc'] = n_words_doc + 1
    model_options['n_words_q'] = n_words_q + 1
    model_options['n_words_ans'] = n_words_ans + 1
    model_options['n_words'] = n_words_q + 1

    print("number words in the answer, ", model_options['n_words_ans'])
    if val_size <= 0 or not val_size:
        valid = SQUADIterator(path=datasets[1],
                              batch_size=batch_size,
                              mode=DataMode.valid,
                              use_pointers=use_pointer_softmax,
                              maxlen=train.maxlen,
                              nwords_doc=n_words_doc,
                              nwords_ans=n_words_ans,
                              assign_priority_point=assign_priority_point,
                              use_doc_emb_ans=use_doc_emb_ans,
                              nwords_q=n_words_q,
                              dicts=[train.doc_dict,
                                    train.q_dict,
                                    train.ans_dict])
    else:
        valid = SQUADIterator(path=datasets[0],
                             batch_size=batch_size,
                             mode=DataMode.valid,
                             use_pointers=use_pointer_softmax,
                             maxlen=train.maxlen,
                             val_idxs=train.val_idxs,
                             nwords_doc=n_words_doc,
                             nwords_ans=n_words_ans,
                             assign_priority_point=assign_priority_point,
                             use_doc_emb_ans=use_doc_emb_ans,
                             nwords_q=n_words_q,
                             dicts=[train.doc_dict,
                                    train.q_dict,
                                    train.ans_dict])

    dictionaries = valid.dicts

    if model_options['use_pointer_softmax']:
        model_options['maxlen'] = train.maxlen

    ensure_dir_exists(dict_dir)

    pkl.dump([train.doc_dict, train.q_dict],
             open(dict_dir + "train_dicts.pkl", "wb"))

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts_r[ii] = dict()
        worddicts[ii] = dictionaries[ii]
        for kk, vv in dd.iteritems():
            worddicts_r[ii][vv] = kk

    if use_pointer_softmax:
        common_words_dict = train.common_words_dict
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

    print('Buliding sampler')
    f_init, f_next = model.build_sampler(tparams, trng, None)

    # before any regularizer
    print('Building f_log_probs...',)

    # Compile the Theano function to compute the log-probs.
    f_log_probs = theano.function(inps, cost, updates=updates, profile=profile)
    print('Done')
    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(q_mask.sum(0) // doc_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print('Building f_cost...', )
    f_cost = theano.function(inps, cost, updates=updates, profile=profile)
    print('Done')
    print('Computing gradient...', )
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print('Done')
    gnorm = tensor.sqrt(sum([(g**2).sum() for g in grads]))

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()

        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2 + 1e-6) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print('Building optimizers...',)

    f_grad_shared, f_update = eval(optimizer)(lr, tparams,
                                              grads, inps,
                                              cost, gnorm, other_updates=updates)

    print('Done')
    print('Optimization')
    history_errs = []

    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])

    best_p = None
    bad_counter = 0
    if validFreq == -1:
        validFreq = len(train[0]) / batch_size

    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    if sampleFreq == -1:
        sampleFreq = len(train[0]) / batch_size

    ave_cost = 0.
    ave_gnorm = 0.
    ave_upd = 0.

    uidx = 0
    estop = False
    max_seqlen = 50

    for eidx in xrange(max_epochs):
        n_samples = 0
        train.reset()
        for batch in train:
            uidx += 1
            use_noise.set_value(1.)

            doc, doc_mask, q, q_mask, ans, ans_locs, ans_mask = batch
            doc, q, ans, ans_locs = prepare_data(doc, doc_mask, q, q_mask, ans,
                                                 ans_locs,
                                                 ans_mask,
                                                 max_len=maxlen)
            n_samples += len(doc)

            if doc is None:
                print('Minibatch with zero sample under length ', maxlen)
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            if use_doc_emb_ans:
                cost, gnorm = f_grad_shared(doc,
                                            doc_mask,
                                            q,
                                            q_mask,
                                            ans_locs,
                                            ans_mask)
            else:
                cost, gnorm = f_grad_shared(doc,
                                            doc_mask,
                                            q,
                                            q_mask,
                                            ans,
                                            ans_mask)

            # do the update on parameters
            up_norm = f_update(lrate)
            ud = time.time() - ud_start

            ave_upd = 0.9 * ave_upd + 0.1 * up_norm
            ave_cost = 0.9 * ave_cost + 0.1 * cost
            ave_gnorm = 0.9 * ave_gnorm + 0.1 * gnorm

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('NaN detected')
                ipdb.set_trace()
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print('Epoch ', eidx, 'Update ', uidx, 'Cost ', ave_cost, 'Up Norm, ', ave_upd, \
                        'G Norm', ave_gnorm, 'UD ', ud)



            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:

                if use_pointer_softmax:
                    print(">>>**str** denotes the words copied from the source.")

                # FIXME: random selection ?
                for jj in xrange(numpy.minimum(5, doc.shape[1])):
                    stochastic = True
                    sample, score = gen_sample(tparams,
                                               f_init,
                                               f_next,
                                               doc[:, jj][:, None],
                                               ans_locs[:, jj][:, None] if use_doc_emb_ans else ans[:, jj][:, None],
                                               doc_mask[:, jj][:, None],
                                               ans_mask[:, jj][:, None],
                                               model_options, trng=trng,
                                               k=1,
                                               maxlen=max_seqlen,
                                               stochastic=stochastic,
                                               argmax=False)
                    try:
                        print()
                        print('Source ', jj, ': ', end=" ")

                        for vv in doc[:, jj]:
                            if vv == worddicts[1]['EOS']:
                                break
                            if vv in worddicts_r[0]:
                                if use_bpe:
                                    print((worddicts_r[0][vv]).replace('@@', ''), end=" ")
                                else:
                                    print(worddicts_r[0][vv], end=" ")
                            else:
                                print('UNK', end=" ")

                        print()
                        print('Truth ', jj, ' : ', end=" ")
                    except:
                        print("Something went wrong and I'm to tired. Sorry.")
                    try:
                        for i, vv in enumerate(q[:, jj]):
                            if q[i:, jj].sum() == 0:
                                break
                            if use_pointer_softmax:
                                if vv >= maxlen:
                                    if vv == maxlen:
                                        break
                                    ovv = vv - maxlen
                                    if ovv in worddicts_r[1]:
                                        print(worddicts_r[1][ovv],
                                              end=" ")
                                    else:
                                        print('UNK',
                                            end=" ")
                                else:
                                    svv = doc[vv, jj]
                                    if svv in worddicts_r[0]:
                                        print(surround(worddicts_r[0][svv]),
                                              end=" ")
                                    else:
                                        print('UNK',
                                              end=" ")
                            else:
                                for vv in q[:, jj]:
                                    if vv == worddicts[1]['EOS']:
                                        break
                                    if vv in worddicts_r[1]:
                                        if use_bpe:
                                            print(worddicts_r[1][vv].replace('@@', ''),
                                                  end=" ")
                                        else:
                                            print(worddicts_r[1][vv],
                                                  end=" ")
                                    else:
                                        print('UNK', end=" ")

                        print("")
                        print('Sample ', jj, ': ', end=" ")
                    except:
                        print("A error occur (Unicode or something). Continuing.")
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]

                    if use_pointer_softmax:
                        for i, vv in enumerate(ss):
                            if numpy.array(ss[i:]).sum() == 0:
                                break

                            if vv >= maxlen:
                                if vv == maxlen:
                                    break

                                ovv = vv - maxlen
                                if ovv in worddicts_r[1]:
                                    try:
                                        print(worddicts_r[1][ovv], end=" ")
                                    except:
                                        print("unicode error")
                                else:
                                    print('UNK', end=" ")

                            else:
                                svv = doc[vv, jj]
                                if svv in worddicts_r[0]:
                                    print(surround(worddicts_r[0][svv]), end=" ")
                                else:
                                    print('UNK', end=" ")

                    else:
                        for vv in ss:
                            if vv == worddicts[1]['EOS']:
                                break
                            if vv in worddicts_r[1]:
                                if use_bpe:
                                    print((worddicts_r[1][vv]).replace('@@', ''), end=" ")
                                else:
                                    print(worddicts_r[1][vv], end=" ")
                            else:
                                print('UNK', end=" ")
                    print("")

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                valid.reset()
                use_noise.set_value(0.)

                valid_errs = pred_probs(f_log_probs,
                                        prepare_data,
                                        model_options,
                                        valid)

                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0

                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():

                    bad_counter += 1
                    if bad_counter > patience:
                        print('Early Stop!')
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()
                print('Valid ', valid_err)

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print('Saving...')

                # if best_p is not None:
                #     params = best_p
                # else:
                params = unzip(tparams)

                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))

                if best_p:
                    print("Saving a new best file!")
                    numpy.savez(saveto+"_best.npz", history_errs=history_errs, **best_p)
                    best_p = None

                print('Done')
            # finish after this many updates
            if uidx >= finish_after:
                print('Finishing after %d iterations!' % uidx)
                estop = True
                break

        print('Seen %d samples' % n_samples)
        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print('Valid ', valid_err)

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                **params)

    return valid_err
