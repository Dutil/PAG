import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from nunits import NSigmoid, NTanh
from model_utils import zipp, unzip, itemlist, prfx, init_tparams, ortho_weight, \
     norm_weight, _slice
from core_d2qgen.utils import dot, sharedX, as_floatX
from core_d2qgen.commons import tanh, leaky_rect, sigmoid, linear, global_rng, global_trng
from core_d2qgen.operators import GumbelSigmoid

#from planning_bak import Planner
import planning

BIG = 100


# batch preparation
def prepare_data(seqs_doc, doc_mask, seqs_q, qmask,
                 seqs_ans=None, seqs_ans_locs=None,
                 ans_mask=None, max_len=None):

    max_ans_len = 0 if ans_mask is None else ans_mask.shape[1]

    seqs_doc_m = numpy.zeros_like(doc_mask)
    seqs_q_m = numpy.zeros_like(qmask)
    seqs_ans_m = numpy.zeros_like(ans_mask)
    seqs_ans_locs_m = numpy.zeros_like(ans_mask)

    for i in xrange(seqs_doc_m.shape[1]):
        if max_len is None:
            max_len = len(seqs_doc[i])

        seqs_doc_m[:len(seqs_doc[i][:max_len]), i] = numpy.array(seqs_doc[i][:max_len])
        seqs_q_m[:len(seqs_q[i]), i] = numpy.array(seqs_q[i])

        if max_ans_len > 0:
            seqs_ans_m[:len(seqs_ans[i]), i] = numpy.array(seqs_ans[i])
            ansl_len = len(seqs_ans[i]) - 1
            seqs_ans_locs_m[:ansl_len, i] = numpy.array(seqs_ans_locs[i][:ansl_len])
        else:
            seqs_ans_m = None
            seqs_ans_locs_m = None

    return seqs_doc_m.astype("uint32"), seqs_q_m.astype("uint32"), \
            seqs_ans_m.astype("uint32"), seqs_ans_locs_m.astype("uint32")


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         state_before * trng.binomial(state_before.shape,
                                                      p=0.5, n=1,
                                                      dtype=state_before.dtype),
                         state_before * 0.5)
    return proj


class Layer(object):

    def __init__(self,
                 prefix=None,
                 options=None,
                 **kwargs):
        self.prefix = prefix
        self.options = options

    def prfx(self, name):
        return '%s_%s' % (self.prefix, name)

    def param_init(self):
        raise NotImplementedError

    def set_p(self,
              params,
              name=None,
              val=None):
        params[self.prfx(name)] = as_floatX(val)

    def get_p(self, params, name):
        return params[self.prfx(name)]


class LayerNorm(Layer):
    def __init__(self,
                 prefix="lnorm",
                 use_bias=True,
                 options=None,
                 **kwargs):
        self.use_bias = use_bias
        super(LayerNorm, self).__init__(prefix, options, **kwargs)

    def param_init(self,
                   params,
                   dim=None,
                   ortho=True):

        if dim is None:
            nout = self.options['dim']

        gamma = numpy.ones((int(dim),)).astype("float32")
        beta = numpy.zeros((int(dim),)).astype("float32")
        self.set_p(params, "gamma", gamma)
        if self.use_bias:
            self.set_p(params, "beta", beta)

    def __call__(self, tparams, state_below, **kwargs):
        #assert state_below.ndim == 2
        if state_below.ndim > 2:
            import pdb; pdb.set_trace()

        mean, var = state_below.mean(axis=1, keepdims=True), \
                state_below.var(axis=1, keepdims=True)

        mean.tag.bn_statistic = True
        mean.tag.bn_label = self.prefix + "_mean"
        var.tag.bn_statistic = True
        var.tag.bn_label = self.prefix + "_var"
        eps = 1e-5
        gamma = self.get_p(tparams, "gamma")
        if self.use_bias:
            beta = self.get_p(tparams, "beta")

        if state_below.ndim > 1:
            gamma = gamma.dimshuffle('x', 0)
            if self.use_bias:
                beta = beta.dimshuffle('x', 0)

        y = (state_below - mean) / tensor.maximum(tensor.sqrt(var + eps), eps)
        y = gamma * y

        if self.use_bias:
            y += beta

        assert y.ndim == 2
        return y


class FFLayer(Layer):

    def __init__(self,
                 prefix='ff',
                 use_bias=True,
                 options=None,
                 **kwargs):
        self.use_bias = use_bias
        super(FFLayer, self).__init__(prefix, options, **kwargs)

    # feed-forward layer: affine transformation + point-wise nonlinearity
    def param_init(self,
                   params,
                   nin=None,
                   nout=None,
                   ortho=True,
                   default_bias=1e-6):

        if nin is None:
            nin = self.options['dim_proj']
        if nout is None:
            nout = self.options['dim_proj']

        W = norm_weight(nin, nout, scale=0.05, ortho=ortho)
        self.set_p(params, 'W', W)

        if self.use_bias:
            b = as_floatX(numpy.random.uniform(high=1e-3, low=-1e-3, size=(nout,)))
            self.set_p(params, 'b', b)
        return params

    def __call__(self,
                tparams,
                state_below,
                activ='linear',
                **kwargs):
        W = self.get_p(tparams, 'W')
        if self.use_bias:
            preact = tensor.dot(state_below, W) + self.get_p(tparams, 'b')
        else:
            preact = tensor.dot(state_below, W)

        return eval(activ)(preact)


def pointer_softmax(alphas, output, switch):
    """
        Pointer softmax
    """
    p1 = alphas * switch
    p2 = output * (1. - switch)
    final_output = tensor.concatenate([p1, p2], axis=-1)
    return final_output


class GRU(Layer):

    def __init__(self, prefix='gru', options=None, **kwargs):
        super(GRU, self).__init__(prefix, options, **kwargs)

    # GRU layer
    def param_init(self,  params, nin=None, dim=None):

        if nin is None:
            nin = self.options['dim_proj']

        if dim is None:
            dim = self.options['dim_proj']

        # embedding to gates transformation weights, biases
        W = numpy.concatenate([norm_weight(nin, dim),
            norm_weight(nin, dim)], axis=1)
        self.set_p(params, 'W', W)

        b = numpy.zeros((2 * dim,)).astype('float32') + 1e-6
        self.set_p(params, 'b', b)

        # recurrent transformation weights for gates
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        self.set_p(params, 'U', U)

        # embedding to hidden state proposal weights, biases
        Wx = norm_weight(nin, dim)
        self.set_p(params, 'Wx', Wx)
        bx = numpy.zeros((dim,)).astype('float32') + 1e-6
        self.set_p(params, 'bx', bx)

        if self.options['use_batch_norm']:
            self.inp_lnorm1 = LayerNorm(use_bias=True,
                                        options=self.options,
                                        prefix=self.prfx("lnorm1"))
            self.inp_lnorm1.param_init(params, dim=2*dim)

            self.inp_lnorm2 = LayerNorm(use_bias=True,
                                        options=self.options,
                                        prefix=self.prfx("lnorm2"))
            self.inp_lnorm2.param_init(params, dim=dim)

            self.inp_lnorm3 = LayerNorm(use_bias=True,
                                        options=self.options,
                                        prefix=self.prfx("lnorm3"))
            self.inp_lnorm3.param_init(params, dim=2*dim)

            self.inp_lnorm4 = LayerNorm(use_bias=True,
                                        options=self.options,
                                        prefix=self.prfx("lnorm4"))
            self.inp_lnorm4.param_init(params, dim=dim)

        # recurrent transformation weights for hidden state proposal
        Ux = ortho_weight(dim)
        self.set_p(params, 'Ux', Ux)
        return params

    def __call__(self,
                 tparams,
                 state_below,
                 mask=None,
                 one_step=False,
                 init_state=None,
                 *args,
                 **kwargs):

        if self.options['use_pointer_softmax']:
            nsteps = self.options['maxlen']
        else:
            nsteps = state_below.shape[0]
            if state_below.ndim == 3:
                n_samples = state_below.shape[1]
            else:
                n_samples = 1

        if state_below.ndim in [2, 3]:
            n_samples = state_below.shape[1]
        elif state_below.ndim == 1:
            if not one_step:
                raise ValueError('if state_below.ndim is 1, one_step shoud also be 1')
        else:
            n_samples = 1

        # mask
        if mask is None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        dim = self.get_p(tparams, 'Ux').shape[1]
        W = self.get_p(tparams, 'W')
        Wx = self.get_p(tparams, 'Wx')
        U = self.get_p(tparams, 'U')
        Ux = self.get_p(tparams, 'Ux')
        b = self.get_p(tparams, 'b')
        bx = self.get_p(tparams, 'bx')

        if mask is None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        if state_below.dtype == 'int64':
            state_below_ = W[state_below.flatten()]
            state_belowx = Wx[state_below.flatten()]

            if state_below.ndim == 2:
                state_below_ = state_below_.reshape((nsteps, n_samples, -1))
                state_belowx = state_belowx.reshape((nsteps, n_samples, -1))

            state_below_ += b
            state_belowx += bx
        else:
            # projected x to hidden state proposal
            state_below_ = tensor.dot(state_below, W) + b
            # projected x to gates
            state_belowx = tensor.dot(state_below, Wx) + bx

        # initial/previous state
        if init_state is None:
            init_state = tensor.alloc(0., n_samples, dim)

        shared_vars = [W, Wx, U, Ux, b, bx]
        if self.options['use_batch_norm']:
            for k, v in tparams.iteritems():
                if "gamma" in k or "beta" in k:
                    shared_vars.append(v)

        # step function to be used by scan
        # arguments    | sequences | outputs-info | non-seqs
        def _step(m_, x_, xx_, h_, *args):
            if self.options['use_batch_norm']:
                x_ = self.inp_lnorm1(tparams, x_)

            if self.options['use_batch_norm']:
                xx_ = self.inp_lnorm2(tparams, xx_)

            preact = dot(h_, U)
            if self.options['use_batch_norm']:
                preact = self.inp_lnorm3(tparams, preact)

            preact += x_
            # reset and update gates
            r = sigmoid(_slice(preact, 0, dim))
            u = sigmoid(_slice(preact, 1, dim))
            # compute the hidden state proposal
            preactx = dot(h_, Ux)
            if self.options['use_batch_norm']:
                preactx = self.inp_lnorm4(tparams, preactx)

            preactx = preactx * r
            preactx = preactx + xx_

            # hidden state proposal
            h = tanh(preactx)

            # leaky integrate and obtain next hidden state
            h = u * h_ + (1. - u) * h
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h

        # prepare scan arguments
        seqs = [mask, state_below_, state_belowx]
        init_states = [init_state]

        if one_step:
            rval = _step(*(seqs + init_states + shared_vars))
        else:
            rval, updates = theano.scan(_step,
                                        sequences=seqs,
                                        outputs_info=init_states,
                                        non_sequences=shared_vars,
                                        name=self.prfx('_layers'),
                                        n_steps=nsteps,
                                        strict=True)
        rval = [rval]
        return rval


class GRUCond(Layer):

    def __init__(self,
                 prefix='gru_cond',
                 options=None,
                 **kwargs):
        super(GRUCond, self).__init__(prefix, options, **kwargs)

    # Conditional GRU layer with Attention
    def param_init(self,
                   params,
                   nin=None,
                   dim=None,
                   dimctx=None,
                   nin_nonlin=None,
                   dim_nonlin=None):

        if nin is None:
            nin = self.options['dim']
        if dim is None:
            dim = self.options['dim']
        if dimctx is None:
            dimctx = self.options['dim']
        if nin_nonlin is None:
            nin_nonlin = nin

        if dim_nonlin is None:
            dim_nonlin = dim

        dim_word = self.options['dim_word']
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim)],
                               axis=1)

        ans_dim = 2*dim if self.options['use_doc_emb_ans'] else dim_word
        Wans = norm_weight(ans_dim, 2*dim).astype("float32")
        self.set_p(params, 'Wans', Wans)
        self.set_p(params, 'W', W)
        self.set_p(params, 'b', numpy.zeros((2 * dim,)).astype('float32'))

        U = numpy.concatenate([ortho_weight(dim_nonlin),
                               ortho_weight(dim_nonlin)], axis=1)
        self.set_p(params, 'U', U)

        Wx = norm_weight(nin_nonlin, dim_nonlin)
        self.set_p(params, 'Wx', Wx)

        Ux = ortho_weight(dim_nonlin)
        self.set_p(params, 'Ux', Ux)
        self.set_p(params, 'bx', numpy.zeros((dim_nonlin,)).astype('float32'))

        U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                                  ortho_weight(dim_nonlin)], axis=1)

        self.set_p(params, 'U_nl', U_nl)
        self.set_p(params, 'b_nl', numpy.zeros((2 * dim_nonlin,)).astype('float32'))

        Ux_nl = ortho_weight(dim_nonlin)
        self.set_p(params, 'Ux_nl', Ux_nl)
        self.set_p(params, 'bx_nl', numpy.zeros((dim_nonlin,)).astype('float32'))

        # context to LSTM
        Wc = norm_weight(dimctx, dim*2)
        self.set_p(params, 'Wc', Wc)

        Wcx = norm_weight(dimctx, dim)
        self.set_p(params, 'Wcx', Wcx)

        # attention: context -> hidden
        Wc_att = norm_weight(dimctx)
        self.set_p(params, 'Wc_att', Wc_att)

        # attention: hidden bias
        b_att = numpy.zeros((dimctx,)).astype('float32')
        self.set_p(params, 'b_att', b_att)

            # attention: combined -> hidden
        W_comb_att = norm_weight(dim, dimctx)
        self.set_p(params, 'W_comb_att', W_comb_att)

        if not self.options['do_planning']:

            # attention:
            U_att = norm_weight(dimctx, 1).astype("float32")
            self.set_p(params, 'U_att', U_att)
            c_att = numpy.zeros((1,)).astype('float32')
            self.set_p(params, 'c_tt', c_att)

        if self.options['use_batch_norm']:
            self.inp_lnorm1 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm1")
            self.inp_lnorm1.param_init(params, dim=2*dim)

            self.inp_lnorm2 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm2")
            self.inp_lnorm2.param_init(params, dim=dim)

            self.inp_lnorm3 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm3")
            self.inp_lnorm3.param_init(params, dim=2*dim_nonlin)

            self.inp_lnorm4 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm4")
            self.inp_lnorm4.param_init(params, dim=dim)

            self.inp_lnorm5 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm5")
            self.inp_lnorm5.param_init(params, dim=dimctx)

            self.inp_lnorm6 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm6")
            self.inp_lnorm6.param_init(params, dim=2*dim_nonlin)

            self.inp_lnorm7 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm7")
            self.inp_lnorm7.param_init(params, dim=2*dim_nonlin)

            self.inp_lnorm8 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm8")
            self.inp_lnorm8.param_init(params, dim=dim_nonlin)

            self.inp_lnorm9 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm9")
            self.inp_lnorm9.param_init(params, dim=dim_nonlin)

            self.inp_lnorm10 = LayerNorm(use_bias=True, options=self.options,
                                        prefix=self.prefix + "lnorm10")
            self.inp_lnorm10.param_init(params, dim=2. * dim_nonlin)


        # planning module
        if self.options['do_planning']:

            self.options['st_estimator'] = "GumbelSoftmax"
            self.options['only_use_w'] = True
            self.options['learn_t'] = True
            self.options['repeat_actions'] = False

            self.planner_options = {'create_param':True, 'repeat_actions':False,
                                    'plan_steps':10, 'ntimesteps':10,
                                    'inter_size':64, 'dec_dim':dim,
                                    'batch_size':None, 'context_dim':dimctx, 'use_gate':True, 'always_recommit':False,
                                    'do_commit':True}

            self.planner = planning.Planner(self.prefix + "planner", self.options, **self.planner_options)
            params.update(self.planner.getParams())


        return params

    def __call__(self,
                 tparams,
                 state_below,
                 ys=None,
                 x_embs=None,
                 mask=None,
                 ans_emb=None,
                 context=None,
                 one_step=False,
                 init_memory=None,
                 init_state=None,
                 context_mask=None,
                 init_commit=None,
                 init_action_plan=None,
                 init_context=None,
                 **kwargs):

        assert context, 'Context must be provided'

        if one_step:
            assert init_state, 'previous state must be provided'

        nsteps = state_below.shape[0]

        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        # mask
        if mask is None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        dim = self.get_p(tparams, 'Wcx').shape[1]

        # initial/previous state
        if init_state is None:
            init_state = tensor.alloc(0., n_samples, dim)

        # projected context
        assert context.ndim == 3, \
            'Context must be 3-d: #annotation x #sample x dim'

        Wc_att = self.get_p(tparams, 'Wc_att')
        bc_att = self.get_p(tparams, 'b_att')

        pctx_ = tensor.dot(context, Wc_att) + bc_att
        if context.ndim == 2:
            context = context.reshape((-1, n_samples, context.shape[-1]))

        pctx_ = pctx_.reshape((-1, n_samples, Wc_att.shape[1]))

        W_ans = self.get_p(tparams, 'Wans')

        ctx_ans = tensor.dot(ans_emb, W_ans)
        if self.options['use_batch_norm']:
            ctx_ans = self.inp_lnorm10(tparams, ctx_ans)

        # projected x
        state_belowx = tensor.dot(state_below, self.get_p(tparams, 'Wx')) + \
            self.get_p(tparams, 'bx')

        state_below_ = tensor.dot(state_below, self.get_p(tparams, 'W')) + \
            self.get_p(tparams, 'b')

        if x_embs and self.options['use_pointer_softmax']:
            state_xembsx = (tensor.dot(x_embs, self.get_p(tparams, 'Wx')) + self.get_p(tparams,
                'bx')).reshape((-1, n_samples, state_belowx.shape[-1]))
            state_xembs = (tensor.dot(x_embs, self.get_p(tparams, 'W')) + self.get_p(tparams,
                'b')).reshape((-1, n_samples, state_below_.shape[-1]))

        U = self.get_p(tparams, 'U')
        Wc = self.get_p(tparams, 'Wc')
        W_comb_att = self.get_p(tparams, 'W_comb_att')

        if not self.options['do_planning']:
            U_att = self.get_p(tparams, 'U_att')
            c_tt = self.get_p(tparams, 'c_tt')
        else:
            U_att = tensor.alloc(1., 1)
            c_tt = tensor.alloc(1., 1)

        Ux = self.get_p(tparams, 'Ux')
        Wcx = self.get_p(tparams, 'Wcx')
        U_nl = self.get_p(tparams, 'U_nl')
        Ux_nl = self.get_p(tparams, 'Ux_nl')
        b_nl = self.get_p(tparams, 'b_nl')
        bx_nl = self.get_p(tparams, 'bx_nl')
        W_ans = self.get_p(tparams, 'Wans')



        shared_vars = [U, Wc,
                       W_comb_att,
                       U_att,
                       c_tt, Ux,
                       Wcx, U_nl,
                       Ux_nl,
                       b_nl, bx_nl,
                       W_ans,
                       Wc_att,
                       bc_att]

        # The planner
        if self.options['do_planning']:
            planner = planning.Planner(self.prefix + "planner", self.options, **self.planner_options)
            planner.setParams(tparams)
            shared_vars += planner.getParams(return_all=True).values()

        if self.options['use_batch_norm']:
            for k, v in tparams.iteritems():
                if "gamma" in k or "beta" in k:
                    shared_vars.append(v)

        ubnorm = self.options['use_batch_norm']
        def _step(m_, x_,
                  xx_, ys,
                  h_, ctx_, alpha_,
                  probs_t, samples_t, probs_origin_t, samples_origin_t, action_plan_t,
                  pctx_, context,
                  *args):

            if ubnorm:
                x_ = self.inp_lnorm1(tparams, x_)
                xx_ = self.inp_lnorm2(tparams, xx_)

            # Define the UNK labels
            preact1 = tensor.dot(h_, U)
            if ubnorm:
                preact1 = self.inp_lnorm3(tparams, preact1)

            preact1 += x_
            preact1 = sigmoid(preact1)

            r1 = _slice(preact1, 0, dim) #+ _slice(ctx_ans, 0, dim)
            u1 = _slice(preact1, 1, dim) #+ _slice(ctx_ans, 1, dim)
            preactx1 = tensor.dot(h_, Ux)

            if ubnorm:
                preactx1 = self.inp_lnorm4(tparams, preactx1)

            preactx1 *= r1
            preactx1 += xx_

            h1 = tanh(preactx1)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

            # attention
            #TO CHANGE

            pstate_ = tensor.dot(h1, W_comb_att)
            if ubnorm:
                pstate_ = self.inp_lnorm5(tparams, pstate_)


            pctx__ = pctx_ + pstate_[None, :, :] + ctx_ans[None, :, :]


            if not self.options['do_planning']:
                probs, samples, commit_origin, probs_origin, alpha, temp = [tensor.alloc(1., 1, 1)] * 6
                action_plan = tensor.alloc(1., 1, 1, 1)
                #pstate_ = tensor.dot(h1, W_comb_att)
                #if ubnorm:
                #    pstate_ = self.inp_lnorm5(tparams, pstate_)
                #pctx__ = pctx_ + pstate_[None, :, :] + ctx_ans[None, :, :]

                pctx__ = tanh(pctx__)
                alpha = tensor.dot(pctx__, U_att) + c_tt
            else:
                params = list(args)
                params = {param.name: param for param in params}

                # commit_origin, probs_origin are the non-shift version of the sampling
                # proj_ctx = proj_ctx_all + state_belowctx_emb_t
                # Should be The context is pctx_ + ctx_ans[None, :, :]?
                #pctx__ = pctx_ + ctx_ans[None, :, :]
                #... h1?

                probs, samples, commit_origin, probs_origin, alpha, action_plan, temp = planner.getAlpha(h1, None,
                                                                                                     pctx__,
                                                                                                     action_plan_t,
                                                                                                     samples_t, probs_t,
                                                                                                     probs_origin_t,
                                                                                                     samples_origin_t,
                                                                                                     params=params,
                                                                                                         context_tm1=ctx_)

            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
            alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
            #mask the alpha's with the context_mask

            if context_mask:
                alpha = alpha * context_mask

            alpha = alpha / (alpha.sum(0, keepdims=True) + 1e-8)

            # current context
            ctx_ = (context * alpha.dimshuffle(0, 1, 'x')).sum(0)

            preact2 = tensor.dot(h1, U_nl) + b_nl

            if ubnorm:
                preact2 = self.inp_lnorm6(tparams, preact2)

            preact2 += tensor.dot(ctx_, Wc)
            if ubnorm:
               preact2 = self.inp_lnorm7(tparams, preact2)

            preact2 = sigmoid(preact2)
            r2 = _slice(preact2, 0, dim)
            u2 = _slice(preact2, 1, dim)

            preactx2 = tensor.dot(h1, Ux_nl) + bx_nl
            if ubnorm:
                preactx2 = self.inp_lnorm8(tparams, preactx2)

            preactx2 *= r2
            preactx2 += tensor.dot(ctx_, Wcx)
            if ubnorm:
                preactx2 = self.inp_lnorm9(tparams, preactx2)

            h2 = tanh(preactx2)
            h2 = u2 * h1 + (1. - u2) * h2
            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

            return h2, ctx_, alpha.T, probs, samples, commit_origin, probs_origin, action_plan

        seqs = [mask, state_below_, state_belowx, ys]

        if self.options['use_pointer_softmax']:
            x_embs_nonseqs = [state_xembs, state_xembsx, ctx_ans]
        else:
            x_embs_nonseqs = [ctx_ans]

        nonseqs = [pctx_, context] + shared_vars + x_embs_nonseqs


        planner_variables = [tensor.alloc(1., 1, 1)] * 4 + [tensor.alloc(1., 1,
                                                                         1, 1)]
        if one_step:

            if self.options['do_planning']:
                planner_variables = [tensor.alloc(1., n_samples, planner.plan_steps),  # probs
                                 init_commit,  # sample
                                 tensor.alloc(1., n_samples, planner.plan_steps),  # sample_origin
                                 tensor.alloc(1., n_samples, planner.plan_steps),  # probs_origin
                                 init_action_plan,  # action_plan
                                 ]

            rval = _step(*(seqs + [init_state, init_context, None] + planner_variables + nonseqs))
            updates = {}
        else:

            if self.options['do_planning']:
                action_plan = tensor.alloc(0., context.shape[0], n_samples, planner.action_plan_steps)
                planner_variables = [tensor.alloc(1., n_samples, planner.plan_steps),  # probs
                                 tensor.alloc(1., n_samples, planner.plan_steps),  # sample
                                 tensor.alloc(1., n_samples, planner.plan_steps),  # sample_origin
                                 tensor.alloc(1., n_samples, planner.plan_steps),  # probs_origin
                                 action_plan,  # action_plan
                                 ]

            rval, updates = theano.scan(_step,
                                        sequences=seqs,
                                        outputs_info=[init_state,
                                                      tensor.alloc(0.,
                                                                   n_samples,
                                                                   context.shape[2]),
                                                      tensor.alloc(0.,
                                                                   n_samples,
                                                                   context.shape[0])] + planner_variables,
                                        non_sequences=nonseqs,
                                        name=prfx(self.prefix, '_layers'),
                                        n_steps=nsteps,
                                        strict=True)
        return rval, updates


class Model(Layer):

    def __init__(self,
                 prefix="da2q",
                 options=None,
                 **kwargs):
        options['use_bias'] = True
        super(Model, self).__init__(prefix, options, **kwargs)
        self.GRUEncoderFwd = GRU("encoder_fwd",
                                 options=options,
                                 **kwargs)
        self.GRUEncoderBwd = GRU("encoder_bwd",
                                 options=options,
                                 **kwargs)
        self.FFState = FFLayer("ff_state",
                               options=options,
                               **kwargs)

        self.FFAns = FFLayer("ff_ans_proj", options=options, **kwargs)
        self.GRUDecoder = GRUCond('decoder', options=options, **kwargs)

        self.FFLogitGRU = FFLayer('ff_logit_gru', options=options, **kwargs)
        self.FFLogitPrev = FFLayer('ff_logit_prev',
                                   use_bias=False,
                                   options=options, **kwargs)
        self.FFLogitCans = FFLayer('ff_logit_cans',
                                   use_bias=False,
                                   options=options, **kwargs)
        self.FFLogitCtx = FFLayer('ff_logit_ctx', options=options, **kwargs)
        self.FFLogit = FFLayer('ff_logit', options=options, **kwargs)
        self.FFSwitchSecond = FFLayer('ff_switch_second',
                                      options=options,
                                      **kwargs)
        self.FFSwitchSingle = FFLayer('ff_switch_single',
                                      options=options,
                                      **kwargs)

    # initialize all parameters
    def init_params(self):
        params = OrderedDict()
        # embedding
        params['Wemb_doc'] = norm_weight(self.options['n_words_doc'],
                                         self.options['dim_word'])
        if not self.options['use_doc_emb_ans']:
            params['Wemb_ans'] = norm_weight(self.options['n_words_ans'],
                                             self.options['dim_word'])

        if self.options['use_pointer_softmax']:
            params['Wemb_dec'] = norm_weight(self.options['n_words_q'] + self.options['maxlen'],
                                             self.options['dim_word'])
        else:
            params['Wemb_dec'] = norm_weight(self.options['n_words_q'],
                                             self.options['dim_word'])

        # encoder: bidirectional RNN
        params = self.GRUEncoderFwd.param_init(params,
                                               nin=self.options['dim_word'],
                                               dim=self.options['dim'])

        params = self.GRUEncoderBwd.param_init(params,
                                               nin=self.options['dim_word'],
                                               dim=self.options['dim'])

        ctxdim = 2 * self.options['dim']

        # init_state, init_cell
        params = self.FFState.param_init(params,
                                         nin=ctxdim,
                                         nout=self.options['dim'])

        # init_state, init_cell
        params = self.FFAns.param_init(params,
                                       nin=ctxdim if self.options['use_doc_emb_ans'] else self.options['dim_word'],
                                       nout=ctxdim)

        # decoder
        params = self.GRUDecoder.param_init(params,
                                            nin=self.options['dim_word'],
                                            dim=self.options['dim'],
                                            dimctx=ctxdim)

        # readout
        params = self.FFLogitGRU.param_init(params,
                                 nin=self.options['dim'],
                                 nout=self.options['dim_word'],
                                 ortho=False)

        params = self.FFLogitPrev.param_init(params,
                                             nin=self.options['dim_word'],
                                             nout=self.options['dim_word'],
                                             ortho=False)

        params = self.FFLogitCans.param_init(params,
                                             nin=ctxdim if self.options['use_doc_emb_ans'] else self.options['dim_word'],
                                             nout=self.options['dim_word'],
                                             ortho=True)

        params = self.FFLogitCtx.param_init(params,
                                            nin=ctxdim,
                                            nout=self.options['dim_word'],
                                            ortho=False)

        params = self.FFLogit.param_init(params,
                                         nin=self.options['dim_word'],
                                         nout=self.options['n_words_q'])

        if self.options['use_pointer_softmax']:
            # From context to switching MLP:
            params['Ux_pt'] = norm_weight(ctxdim,
                                        self.options['dim']).astype("float32")

            # Switching MLP bias:
            params['b_pt'] = numpy.zeros((self.options['dim'],)).astype("float32")

            # Hidden state of the decoder to the Switching MLP:
            params['Ws_pt'] = norm_weight(self.options['dim'],
                                        self.options['dim']).astype("float32")

            params = self.FFSwitchSecond.param_init(params,
                                                    nin=self.options['dim'],
                                                    nout=self.options['dim'])

            params = self.FFSwitchSingle.param_init(params,
                                                    nin=self.options['dim'],
                                                    default_bias=-1.2,
                                                    nout=1)
        return params

    # build a training model
    def build_model(self, tparams, valid):
        opt_ret = dict({})

        trng = RandomStreams(1234)
        use_noise = sharedX(0., name="use_noise")

        # description string: #words x #samples
        doc = tensor.matrix('doc', dtype='int64')
        doc_mask = tensor.matrix('doc_mask', dtype='float32')
        q = tensor.matrix('q', dtype='int64')
        q_mask = tensor.matrix('q_mask', dtype='float32')
        ans = tensor.matrix('a', dtype="int64")
        ans_mask = tensor.matrix('a_mask', dtype="float32")

        if self.options['debug']:
            vdoc, vdoc_mask, vq, vqmask, vans, vans_locs, vans_mask = next(valid)
            valid.reset()
            d_, q_, a_, ans_locs = prepare_data(vdoc,
                                      vdoc_mask,
                                      vq, vqmask,
                                      vans,
                                      vans_locs,
                                      vans_mask)

            theano.config.compute_test_value = "warn"
            doc.tag.test_value = d_
            doc_mask.tag.test_value = vdoc_mask
            q.tag.test_value = q_
            q_mask.tag.test_value = vqmask

            if not self.options['use_doc_emb_ans']:
                ans.tag.test_value = a_
            else:
                ans.tag.test_value = ans_locs

            ans_mask.tag.test_value = vans_mask

        # for the backward rnn, we just need to invert x and x_mask
        docr = doc[::-1]
        docr_mask = doc_mask[::-1]

        n_timesteps = doc.shape[0]
        n_timesteps_trg = q.shape[0]
        n_samples = doc.shape[1]

        # word embedding for forward rnn (source)
        doc_flat = doc.flatten()
        doc_emb = tparams['Wemb_doc'][doc_flat]
        doc_emb = doc_emb.reshape([n_timesteps,
                                   n_samples,
                                   -1])

        proj = self.GRUEncoderFwd(tparams,
                                  doc_emb,
                                  mask=doc_mask)

        # word embedding for backward rnn (source)
        docr_flat = docr.flatten()
        doc_embr = tparams['Wemb_doc'][docr_flat]
        doc_embr = doc_embr.reshape([n_timesteps,
                                    n_samples,
                                    -1])

        projr = self.GRUEncoderBwd(tparams,
                                   doc_embr,
                                   mask=docr_mask)

        # context will be the concatenation of forward and backward RNNs
        ctx = tensor.concatenate([proj[0],
                                  projr[0][::-1]],
                                  axis=proj[0].ndim - 1)

        # mean of the context (across time) will be used to initialize decoder RNN
        ctx_mean = (ctx * doc_mask[:, :, None]).sum(0) \
                / doc_mask.sum(0)[:, None]

        ans_flat = ans.flatten()

        if self.options['use_doc_emb_ans']:
            idxs = (tensor.arange(ctx.shape[0]*ctx.shape[1], step=ctx.shape[0])[None, :] +
                    ans).flatten()
            cans_mask = doc_mask.flatten()[idxs].reshape(idxs.shape)
            ctx_ans_sel = ctx.reshape((-1, ctx.shape[-1]))[idxs].reshape((-1, ctx.shape[1],
                ctx.shape[2]))
            ctx_ans = ((ans_mask.dimshuffle(0, 1, 'x') * ctx_ans_sel).sum(0) /
                    (ans_mask.sum(axis=0).dimshuffle(0, 'x') + 1e-6))
        else:
            ans_emb = tparams['Wemb_ans'][ans_flat]
            ans_emb = ans_emb.reshape((ans.shape[0], ans.shape[1], -1))
            ctx_amean = (ans_mask.dimshuffle(0, 1, 'x') * ans_emb).sum(0) / ans_mask.sum(axis=0).dimshuffle(0, 'x')
            ctx_ans = tanh(ctx_amean)

        ctx_ans_proj = self.FFAns(tparams,
                                  ctx_ans,
                                  activ='linear')

        ctx_mean += tanh(ctx_ans_proj + ctx_ans)

        # or you can use the last state of
        # forward + backward encoder RNNs
        # initial decoder state
        init_state = self.FFState(tparams,
                                  ctx_mean,
                                  activ='tanh')

        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        q_flat = q.flatten()
        emb = tparams['Wemb_dec'][q_flat]
        emb = emb.reshape([n_timesteps_trg,
                           n_samples,
                          -1])

        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:],
                                           emb[:-1])
        emb = emb_shifted

        # decoder - pass through the decoder conditional gru with attention
        proj, updates = self.GRUDecoder(tparams,
                               emb,
                               ys=q,
                               x_embs=doc_emb,
                               ans_emb=ctx_ans,
                               mask=q_mask,
                               context=ctx,
                               context_mask=doc_mask,
                               one_step=False,
                               init_state=init_state)

        # hidden states of the decoder GRU
        proj_h = proj[0]

        # weighted averages of context,
        # generated by attention module
        ctxs = proj[1]
        alphas = proj[2]

        # weights (alignment matrix)
        opt_ret['dec_alphas'] = alphas

        # compute word probabilities
        logit_lstm = self.FFLogitGRU(tparams,
                                     proj_h,
                                     activ='linear')

        # logit_prev
        logit_prev = self.FFLogitPrev(tparams, emb,
                                      activ='linear')

        # logit_ctx
        logit_ctx = self.FFLogitCtx(tparams, ctxs,
                                     activ='linear')

        logit_cans = self.FFLogitCans(tparams, ctx_ans,
                                      activ='linear')
        shp = (-1, logit_cans.shape[0], logit_cans.shape[1])

        # logit layer
        logit = tanh(logit_lstm.reshape(shp) + logit_prev.reshape(shp) + logit_ctx.reshape(shp) +\
                     logit_cans.dimshuffle('x', 0, 1))

        if self.options['use_dropout']:
            logit = dropout_layer(logit, use_noise, trng)

        logit = self.FFLogit(tparams,
                             logit,
                             activ='linear')

        logit_shp = [proj_h.shape[0], proj_h.shape[1], logit.shape[-1]]
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
                                                   logit_shp[2]]))

        if self.options['use_pointer_softmax']:
            proj_h = proj_h.reshape((proj_h.shape[0] * proj_h.shape[1],
                                    -1))

            ctxs = ctxs.reshape((ctxs.shape[0] * ctxs.shape[1], -1))
            alphas = alphas.reshape((alphas.shape[0] * alphas.shape[1], -1))

            act = NTanh
            switch_l1 =  act(tensor.dot(proj_h, tparams['Ws_pt']) + \
                                        tensor.dot(ctxs, tparams['Ux_pt']) + \
                                        tparams['b_pt'])

            switch_l2 = act(self.FFSwitchSecond(tparams,
                                            switch_l1,
                                            activ='linear'))

            gsigmoid = GumbelSigmoid(itemp=3.0)
            switch_l2 += switch_l1

            temp = self.options['temp_switch']
            swtch_pre = self.FFSwitchSingle(tparams,
                                            switch_l2,
                                            activ='linear') + 0.9

            swtch = gsigmoid(swtch_pre, trng)

            # Check the shapes to see whether if they are correct or not.
            assert swtch.ndim == 2
            swtch = tensor.addbroadcast(swtch, 1)
            probs = pointer_softmax(alphas, probs, swtch)

        if self.options['use_pointer_softmax']:
            y_selm = tensor.cast(tensor.eq(q, doc_emb.shape[0] + 1),
                                        "int64")

            loc_vals = alphas.argmax(-1).reshape((q.shape[0], q.shape[1]))
            q_ = y_selm * loc_vals + (1 - y_selm) * q
        else:
            q_ = q

        # cost
        q_flat = q_.flatten()

        identity = tensor.eye(probs.shape[-1], probs.shape[-1]).astype("float32")
        ans_sum = identity[ans.flatten()].reshape((ans.shape[0], ans.shape[1],
            probs.shape[-1])).sum(0)

        # Check this again!!!
        if not self.options['use_pointer_softmax']:
            tot_size = self.options['n_words_q']
            q_flat_idx = tensor.arange(q_flat.shape[0]) * (self.options['n_words_q']) + q_flat
        else:
            tot_size = self.options['n_words_q'] + self.options['maxlen']
            q_flat_idx = tensor.arange(q_flat.shape[0]) * tot_size + q_flat

        ans_cost = (tensor.log(probs + 1e-8).reshape((q.shape[0], q.shape[1], -1)) * ans_sum.dimshuffle('x', 0, 1)).sum(0)
        cost = -tensor.log(probs.flatten()[q_flat_idx] + 1e-8)
        cost = cost.reshape([q.shape[0], q.shape[1]])
        cost = (cost * q_mask).sum(0) + (self.options['ans_cost_lambda'] * ans_cost.sum(-1) /
                ans_mask.sum(0))

        return trng, use_noise, doc, doc_mask, q, q_mask, ans, ans_mask, opt_ret, cost, updates

    # build a sampler
    def build_sampler(self, tparams, trng, valid=None):

        doc = tensor.matrix('doc', dtype='int64')
        ans = tensor.matrix('ans', dtype='int64')
        q = tensor.vector('q_sampler', dtype='int64')
        doc_mask = tensor.matrix('doc_mask', dtype='float32')
        ans_mask = tensor.matrix('ans_mask', dtype='float32')

        # x: 1 x 1
        init_state = tensor.matrix('init_state', dtype='float32')

        if self.options['debug'] and False:
            if not valid:
                raise ValueError("Validation iterator should not be empty!")

            vdoc, vdoc_mask, vq, vqmask, vans, vans_locs, vans_mask = valid.next()
            valid.reset()

            d_, q_, a_, ans_locs = prepare_data(vdoc,
                                      vdoc_mask,
                                      vq, vqmask, vans,
                                      vans_locs,
                                      vans_mask)

            theano.config.compute_test_value = "warn"
            doc.tag.test_value = d_
            q.tag.test_value = q_

            if not self.options["use_doc_emb_ans"]:
                ans.tag.test_value = a_
            else:
                ans.tag.test_value = ans_locs

            ans_mask.tag.test_value = vans_mask
            doc_mask.tag.test_value = vdoc_mask

            init_state.tag.test_value = numpy.zeros((d_.shape[1],
                                                    self.options['dim'])).astype("int64")

        docr = doc[::-1]
        docr_mask = doc_mask[::-1]

        n_timesteps = doc.shape[0]
        n_samples = doc.shape[1]

        # word embedding (source), forward and backward
        doc_emb = tparams['Wemb_doc'][doc.flatten()]
        doc_emb = doc_emb.reshape([n_timesteps, n_samples, -1])
        doc_embr = tparams['Wemb_doc'][docr.flatten()]
        doc_embr = doc_embr.reshape([n_timesteps, n_samples, -1])

        proj = self.GRUEncoderFwd(tparams,
                                  doc_emb,
                                  mask=doc_mask)

        projr = self.GRUEncoderBwd(tparams,
                                   doc_embr,
                                   mask=docr_mask)

        # concatenate forward and backward rnn hidden states
        ctx = tensor.concatenate([proj[0],
                                projr[0][::-1]],
                                axis=proj[0].ndim-1)


        if not self.options['use_doc_emb_ans']:
            W_ans = tparams['Wemb_ans']
            ans_flat = ans.flatten()
            ans_emb = W_ans[ans_flat]
            ans_emb = ans_emb.reshape((ans.shape[0], ans.shape[1], -1))
            ctx_amean = (ans_mask.dimshuffle(0, 1, 'x') * ans_emb).sum(0) / ans_mask.sum(axis=0).dimshuffle(0, 'x')

            ctx_ans = tanh(ctx_amean)
        else:
            idxs = (tensor.arange(ctx.shape[0]*ctx.shape[1], step=ctx.shape[0])[None, :] +
                    ans).flatten()
            ctx_ans_sel = ctx.reshape((-1, ctx.shape[-1]))[idxs].reshape((-1, ctx.shape[1],
                ctx.shape[2]))
            ctx_ans = ((ans_mask.dimshuffle(0, 1, 'x') * ctx_ans_sel).sum(0) /
                    (ans_mask.sum(axis=0).dimshuffle(0, 'x') + 1e-6))


        ctx_ans_proj = self.FFAns(tparams,
                                  ctx_ans,
                                  activ='linear')

        # get the input for decoder rnn initializer MLP
        ctx_mean = ctx.mean(0)
        ctx_mean += tanh(ctx_ans_proj + ctx_ans)

        init_state = self.FFState(tparams,
                                  ctx_mean,
                                  activ='tanh')

        #For the planner
        init_commit = tensor.matrix('init_commit', dtype='float32')
        init_context = tensor.matrix('init_context', dtype='float32')
        init_action_plan = tensor.tensor3('init_action_plan', dtype='float32')

        print 'Building f_init...',
        outs = [init_state, ctx, doc_emb]
        f_init = theano.function([doc, ans, doc_mask, ans_mask], outs, name='f_init')
        print 'Done'

        # if it's the first word, emb should be
        # all zero and it is indicated by -1
        emb = tensor.switch(q[:, None] < 0,
                            tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                            tparams['Wemb_dec'][q])

        # apply one step of conditional gru with attention
        proj, updates = self.GRUDecoder(tparams,
                               emb,
                               x_embs=doc_emb,
                               ans_emb=ctx_ans,
                               ys=q,
                               mask=None,
                               context=ctx,
                               context_mask=doc_mask,
                               one_step=True,
                               init_state=init_state,
                               init_commit=init_commit, init_action_plan=init_action_plan, init_context=init_context)

        # get the next hidden state
        next_state = proj[0]

        # get the weighted averages of context for this target word y
        ctxs = proj[1]
        alphas = proj[2]

        next_commit = proj[4]
        next_action_plan = proj[7]

        # compute word probabilities
        logit_lstm = self.FFLogitGRU(tparams,
                                     next_state,
                                     activ='linear')

        # logit_prev
        logit_prev = self.FFLogitPrev(tparams, emb,
                                      activ='linear')

        # logit_ctx
        logit_ctx = self.FFLogitCtx(tparams, ctxs,
                                    activ='linear')

        logit_cans = self.FFLogitCans(tparams, ctx_ans,
                                      activ='linear')

        logit = tanh(logit_lstm + logit_prev + logit_ctx + logit_cans)
        logit = self.FFLogit(tparams,
                             logit,
                             activ='linear')

        # compute the softmax probability
        soft_probs = tensor.nnet.softmax(logit)

        if self.options['use_pointer_softmax']:
            act = NTanh
            # Check the shapes to see whether if they are correct or not.
            switch_l1 = act(tensor.dot(next_state,
                                tparams['Ws_pt']) + \
                                tensor.dot(ctxs,
                                        tparams['Ux_pt']) + \
                                        tparams['b_pt'])

            switch_l2 = act(self.FFSwitchSecond(tparams,
                                            switch_l1,
                                            activ='linear') + switch_l1) + 0.9

            gsigmoid = GumbelSigmoid(itemp=3.2)
            temp = self.options['temp_switch']
            swtch_pre = self.FFSwitchSingle(tparams,
                                            switch_l2,
                                            activ='linear')

            swtch = gsigmoid(swtch_pre, trng)
            swtch = tensor.addbroadcast(swtch, 1)
            next_probs = pointer_softmax(alphas, soft_probs, swtch)
        else:
            next_probs = soft_probs

        # sample from softmax distribution to get the sample
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)
        next_sample = next_sample.flatten()

        # compile a function to do the whole thing above, next word probability,
        # sampled word for the next target, next hidden state to be used
        print 'Building f_next..',

        #TODO: Add the answer related changes here.
        inps = [q, ctx, init_state, doc_emb, ans, doc_mask, ans_mask, init_commit, init_action_plan, init_context]
        outs = [next_probs, next_sample, next_state, next_commit, next_action_plan, ctxs]

        f_next = theano.function(inps,
                                outs,
                                updates=updates,
                                name='f_next',
                                on_unused_input="warn")
        print 'Done'
        return f_init, f_next
