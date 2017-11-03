import numpy
np=numpy

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import scipy
import scipy.linalg
from scipy.linalg import circulant

from core.commons import EPS, sigmoid, tanh, floatX, HardSigmoid
from core.utils import concatenate, get_key_byname_from_dict, sharedX, \
                                as_floatX, block_gradient


class Operator(object):
    def __init__(self, eps=1e-8):
        if eps is None:
            self.eps = EPS
        else:
            self.eps = eps


class CosineSimilarity(Operator):
    """
    Computes the cosine similarity between two theano tensor variables.
    """
    def __call__(self, x, y):
        assert x.ndim == y.ndim, "The number of dims for x and y should be equal."
        res = None
        if x.ndim == 3:
            nume = (x*y).sum(2)
            denom = x.norm(2, axis=2) * x.norm(2, axis=2)
            res = nume / (denom + self.eps)
        else:
            nume = (x*y).sum(1)
            denom = x.norm(2, axis=1) * x.norm(2, axis=1)
            res = nume / (denom + self.eps)
        return res


class MemorySimilarity(Operator):

    """
    Computes the cosine similarity between two theano tensor variables.
    """

    def __call__(self, key, M, use_deno=True):
        res = None
        if key.ndim == 3:
            key = key.dimshuffle(0, 1, 'x', 2)
            M = M.dimshuffle('x', 'x', 0, 1)
            key = key / TT.maximum(TT.sqrt(TT.sqr(key).sum(axis=3, keepdims=True)), 1e-5)
            M = M / TT.maximum(TT.sqrt(TT.sqr(M).sum(axis=3, keepdims=True)), 1e-5)
            res = (key * M).sum(3)
        elif key.ndim == 2:
            key = key.dimshuffle(0, 'x', 1)
            if M.ndim == 2:
                M = M.dimshuffle('x', 0, 1)
            key = (key + self.eps) / TT.maximum(TT.sqrt(TT.sqr(key).sum(axis=2, keepdims=True)), 1e-4)
            M = (M + self.eps)/ TT.maximum(TT.sqrt(TT.sqr(M).sum(axis=2, keepdims=True)), 1e-4)
            res = (key * M).sum(2)
        else:
            raise ValueError("The number of dimensions for the key should be"
                             "greater than 1 in MemorySimilarity class")
        return res


class GeomEuclideanSigmoidDot(Operator):
    """
    Computes the geometric average of euclidean distance and sigmoid dot product.
    """
    def __call__(self, key, M, alpha=1, c=1):
        sim = 0
        if key.ndim == 3:
            key = key.dimshuffle(0, 1, 'x', 2)
            M = M.dimshuffle('x', 'x', 0, 1)
            euclid_dist = 1 / (1 + TT.sqrt(((key - M)**2).sum(3)))
            sigm_prod = TT.nnet.sigmoid((key * M).sum(3) + c)
            sim = 0.5 * (alpha**2 * (euclid_dist * sigm_prod))
        else:
            key = key.dimshuffle(0, 'x', 1)
            if M.ndim == 2:
                M = M.dimshuffle('x', 0, 1)
            euclid_dist = 1 / (1 + TT.sqrt(((key - M)**2).sum(2)))
            sigm_prod = TT.nnet.sigmoid((key * M).sum(2) + c)
            sim = 0.5 * (alpha**2 * (euclid_dist * sigm_prod))
        return sim

class CircularConvolve(Operator):

    """
        Perform the circular convolution.
    """
    def __call__(self, weights, shifts, mem_size, shift_width=3):

        if shifts is None or weights is None or mem_size is None:
            raise ValueError("signals, memory sizes and shift_width should not be empty.")

        shift_conv = scipy.linalg.circulant(np.arange(mem_size)).T[np.arange(-int(shift_width / 2),
                                                                              int(shift_width / 2) + 1)][::-1]
        sum_ax = 1
        if weights.ndim == 2:
            W_aux = TT.zeros((shift_width, weights.shape[0], weights.shape[1]))
            for i in xrange(shift_width):
                W_aux = TT.set_subtensor(W_aux[i], weights[:, shift_conv[i]])
            W = W_aux.dimshuffle(1, 0, 2)
        elif weights.ndim == 3:
            sum_ax = 2
            W_aux = TT.zeros((shift_width, weights.shape[0], weights.shape[1],
                              weights.shape[2]))
            for i in xrange(shift_width):
                W_aux = TT.set_subtensor(W_aux[i], weights[:, :, shift_conv[i]])
            W = W_aux.dimshuffle(1, 2, 0, 3)

        result = (shifts * W).sum(axis=sum_ax)
        return result


class CircularConvolveAdvIndexing(Operator):

    """
        Perform the circular convolution.
    """
    def __call__(self, weights, shifts, mem_size, shift_width=3):

        if shifts is None or weights is None or mem_size is None:
            raise ValueError("signals, memory sizes and shift_width should not be empty.")

        shift_conv = scipy.linalg.circulant(np.arange(mem_size)).T[np.arange(-int(shift_width / 2),
                                                                              int(shift_width / 2) + 1)][::-1]
        sum_ax = 1
        if weights.ndim == 2:
            W = weights[:, shift_conv]
        elif weights.ndim == 3:
            sum_ax = 2
            W = weights[:, :, shift_conv]

        result = (shifts * W).sum(axis=sum_ax)
        return result


class CircularConvolveAll(Operator):
    """
        Perform the circular convolution.
    """
    def __call__(self, shifts, weights, mem_size, shift_width=3):

        if shifts is None or weights is None or mem_size is None or shift_width is None:
            raise ValueError("signals, memory sizes and shift_width should not be empty.")

        shift_conv = scipy.linalg.circulant(np.arange(mem_size))

        if weights.ndim == 2:
            S = shifts[:, shift_conv]
        elif weights.ndim == 3:
            S = shifts[:, :, shift_conv]

        sum_ax = 1
        if weights.ndim == 2 and S.ndim - weights.ndim == 1:
            sum_ax = 1
            weights = weights.dimshuffle(0, 'x', 1)
        elif weights.ndim == 2 and S.ndim - weights.ndim == 2:
            sum_ax = 2
            weights = weights.dimshuffle(0, 'x', 'x', 1)

        result = (weights * S).sum(axis=sum_ax)
        return result


class REINFORCE(Operator):

    def __init__(self,
                 lambda1_reg=None,
                 lambda2_reg=None,
                 use_rms_baseline=False,
                 schedule_h_opts=None,
                 generative_pred=True,
                 eps=1e-6,
                 maxlen=16,
                 use_cost_std=True,
                 decay=0.9):

        self.lambda1_reg = lambda1_reg
        self.lambda2_reg = lambda2_reg
        self.use_rms_baseline = use_rms_baseline
        self.decay = decay
        self.use_cost_std = use_cost_std
        self.maxlen = maxlen
        self.generative_pred = generative_pred

        if not schedule_h_opts:
            schedule_h_opts = {}
            schedule_h_opts["lambda2_reg_start"] = 1e-4
            schedule_h_opts["end_nbatches"] = 1000

        self.schedule_h_opts = schedule_h_opts
        self.eps = eps
        self.updates = {}

    def __call__(self, probs,
                 samples,
                 updates,
                 cost=None,
                 mask=None,
                 deterministic=False,
                 child_probs=None,
                 dimshuffle_probs=False,
                 child_samples=None):

        if input is None:
            raise ValueError("input for the %s should "
                             " not be empty." % __class__.__name__)

        key_baseline = get_key_byname_from_dict(updates, "baseline")
        step = 0

        if key_baseline:
            rbaseline = updates[key_baseline]
            key_step = get_key_byname_from_dict(updates, "step")
            if key_step:
                step = updates[key_step]
            else:
                step = sharedX(0., name="step")
        else:
            if self.generative_pred:
                baseline = sharedX(np.zeros((self.maxlen,)) + 1.0 + self.eps, name="baseline")
            else:
                baseline = sharedX(0. + 1.0 + self.eps, name="new_baseline")

            key_step = get_key_byname_from_dict(updates, "step")
            fix_decay = self.decay**(step + as_floatX(1))

            if key_step:
                step = updates[key_step]
            else:
                step = sharedX(0., name="step")
                updates[step] = step + as_floatX(1)

            if self.use_rms_baseline:
                if self.generative_pred:
                    new_baseline = as_floatX(self.decay) * baseline + as_floatX(1 - self.decay) * cost.mean(-1)**2
                else:
                    new_baseline = as_floatX(self.decay) * baseline + as_floatX(1 - self.decay) * cost.mean()**2

                updates[baseline] = new_baseline
                rbaseline = new_baseline / (1 - fix_decay)
                rbaseline = TT.sqrt(rbaseline)
            else:
                if self.generative_pred:
                    new_baseline = as_floatX(self.decay) * baseline + as_floatX(1 - self.decay) * cost.mean(-1)
                else:
                    new_baseline = as_floatX(self.decay) * baseline + as_floatX(1 - self.decay) * cost.mean()

                updates[baseline] = new_baseline
                rbaseline = new_baseline

        key_cvar = get_key_byname_from_dict(updates, "cost_var")

        if key_cvar:
            cost_var = updates[key_cvar]
            new_cost_var = cost_var
        else:
            if self.generative_pred:
                cost_var = sharedX(np.zeros((self.maxlen,)) + as_floatX(1.2), name="cost_var")
                cost_var_ave = (cost.mean(-1) - new_baseline)**2
            else:
                cost_var = sharedX(as_floatX(1.2), name="cost_var")
                cost_var_ave = (cost.mean() - new_baseline)**2

            new_cost_var = as_floatX(self.decay) * cost_var + as_floatX(1 - self.decay) * cost_var_ave
            updates[cost_var] = new_cost_var

        lambda2_reg = self.lambda2_reg

        """
        if not self.schedule_h_opts:
            start = self.schedule_h_opts["lambda2_reg_start"]
            nbatches = self.schedule_h_opts["end_nbatches"]
            end = self.lambda2_reg
            assert start > end
            lambda2_reg = TT.minimum(((start - end) * step / nbatches) + start,
                                       end)
        """

        if dimshuffle_probs:
            probsd = probs.dimshuffle(0, 2, 1)
        else:
            probsd = probs

        if probs.ndim == 3 and cost.ndim == 1:
            if dimshuffle_probs:
                reward = cost.dimshuffle('x', 'x', 0)
                if self.generative_pred:
                    rbaseline = rbaseline.dimshuffle('x', 'x', 0)
                    cost_std = new_cost_var.dimshuffle('x', 'x', 0)
            else:
                reward = cost.dimshuffle('x', 0, 'x')
                if self.generative_pred:
                    rbaseline = rbaseline.dimshuffle('x', 0, 'x')
                    cost_std = new_cost_var.dimshuffle('x', 0, 'x')
        elif probs.ndim == 3 and cost.ndim == 2:
            if dimshuffle_probs:
                reward = cost.dimshuffle(0, 'x', 1)
                if self.generative_pred:
                    rbaseline = rbaseline.dimshuffle(0, 'x', 'x')
                    new_cost_var = new_cost_var.dimshuffle(0, 'x', 'x')
            else:
                reward = cost.dimshuffle('x', 0, 1)
                if self.generative_pred:
                    rbaseline = rbaseline.dimshuffle('x', 0, 'x')
                    new_cost_var = new_cost_var.dimshuffle('x', 0, 'x')

        elif probs.ndim == 4 and self.cost.ndim == 1:
            reward = cost.dimshuffle('x', 'x', 0, 'x')
        elif probs.ndim == 4:
            reward = cost.dimshuffle(0, 'x', 1, 'x')

        centered_cost = reward - rbaseline
        N = probsd.shape[-1]
        if self.use_cost_std:
            cost_std = TT.maximum(TT.sqrt(new_cost_var + 1e-8), 1.0)
        else:
            cost_std = 1

        if child_probs is not None and child_samples is not None:
            cprobs1 = child_samples / (child_probs + 1e-8) + samples / (probsd + 1e-8)
        else:
            cprobs1 = samples / (probsd + 1e-8)

        gradp = self.lambda1_reg * (centered_cost / cost_std) * \
                (cprobs1) + (lambda2_reg) * (TT.log(probsd + 1e-8) + as_floatX(1))

        if dimshuffle_probs:
            gradp = gradp.dimshuffle(0, 2, 1)

        if mask is not None:
            if dimshuffle_probs:
                gradp = mask.dimshuffle(0, 1, 'x') * gradp
            else:
                gradp = mask.dimshuffle(0, 1, 'x') * gradp / N

        known_grads = {probs: gradp}
        policy = -(TT.log(probsd + 1e-8) * samples).mean((1, 2)).sum()
        return updates, known_grads, rbaseline, cost_std, policy, lambda2_reg


class REINFORCEBaselineExt(Operator):

    def __init__(self,
                 lambda1_reg,
                 lambda2_reg,
                 use_rms_baseline=False,
                 schedule_h_opts=None,
                 generative_pred=False,
                 eps=1e-8,
                 decay=0.9,
                 maxlen=16):

        assert lambda1_reg is not None
        assert lambda2_reg is not None

        self.lambda1_reg = lambda1_reg
        self.lambda2_reg = lambda2_reg
        self.use_rms_baseline = use_rms_baseline
        self.decay = decay
        self.maxlen = maxlen
        self.generative_pred = generative_pred

        if not schedule_h_opts:
            schedule_h_opts = {}
            schedule_h_opts["lambda2_reg_start"] = 3e-2
            schedule_h_opts["end_nbatches"] = 4000

        self.schedule_h_opts = schedule_h_opts
        self.eps = eps
        self.updates = {}

    """
        Perform the dropout on the layer.
    """
    def __call__(self, probs,
                 samples,
                 baseline,
                 updates,
                 cost = None,
                 cost_mean=None,
                 mask=None,
                 seq_len=20,
                 batch_size=140,
                 deterministic=False,
                 dimshuffle_probs=True):

        print("Using the input based baseline")
        if input is None:
            raise ValueError("input for the %s should"
                             " not be empty." % __class__.__name__)

        if cost_mean is None:
            cost_mean = cost.mean()

        step = 0
        key_step = get_key_byname_from_dict(updates, "step")
        if key_step:
            step = updates[key_step]
        else:
            step = sharedX(0., name="step")
            updates[step] = step + as_floatX(1)

        key_center = get_key_byname_from_dict(updates, "center")
        if key_center:
            center = updates[key_center]
            new_center = center
        else:
            if self.generative_pred:
                center = sharedX(np.zeros((self.maxlen,)) + 0.15 + self.eps, name="center")
                new_center = as_floatX(self.decay) * center + as_floatX(1 - self.decay) * cost.mean(-1)
            else:
                center = sharedX(0.15 + self.eps, name="center")
                assert cost_mean is not None, "Cost mean should not be empty!"
                if cost.ndim > 2 and cost.broadcastable[0] is False:
                    new_center = as_floatX(self.decay) * center + as_floatX(1 - self.decay) * cost_mean
                else:
                    new_center = as_floatX(self.decay) * center + as_floatX(1 - self.decay) * cost_mean
            updates[center] = new_center

        key_cvar = get_key_byname_from_dict(updates, "cost_var")
        if key_cvar:
            cost_var = updates[key_cvar]
            new_cost_var = cost_var
        else:
            if self.generative_pred:
                cost_var_tot = (cost_mean - new_center)**2
                cost_var = sharedX(numpy.zeros((self.maxlen,)) + as_floatX(1.0), name="cost_var")
            else:
                if cost.ndim >  2 and cost.broadcastable[0] is False:
                    cost_var_tot = (cost_mean - new_center)**2
                else:
                    cost_var_tot = (cost_mean - new_center)**2

                cost_var = sharedX(1.0, name="cost_var")

            new_cost_var = as_floatX(self.decay) * cost_var + as_floatX(1 - self.decay) * \
                    cost_var_tot
            updates[cost_var] = new_cost_var

        lambda2_reg = self.lambda2_reg
        """
        if not self.schedule_h_opts:
            start = self.schedule_h_opts["lambda2_reg_start"]
            nbatches = self.schedule_h_opts["end_nbatches"]
            end = self.lambda2_reg
            assert start > end
            lambda2_reg = TT.minimum(((start - end) * step / nbatches) + start,
                                       end)
        """

        if dimshuffle_probs:
            probsd = probs.dimshuffle(0, 2, 1)
        else:
            probsd = probs

        if samples.ndim == 4:
            reward = cost.dimshuffle(0, 'x', 1, 'x')
            policy = -(TT.log(probsd + 1e-8) * samples).mean((2, 3)).sum()
        else:
            if cost.ndim == 2:
                if dimshuffle_probs:
                    reward = cost.dimshuffle(0, 'x', 1)
                    if self.generative_pred:
                        new_center = new_center.dimshuffle(0, 'x', 'x')
                        new_cost_var = new_cost_var.dimshuffle(0, 'x', 'x')
                    baseline = baseline.dimshuffle(0, 2, 1)
                else:
                    reward = cost.dimshuffle(0, 1, 'x')
                policy = -(TT.log(probsd + 1e-8) * samples).mean((1, 2)).sum()
            elif cost.ndim == 1:
                reward = cost.dimshuffle('x', 0, 'x')
                if dimshuffle_probs:
                    baseline = baseline.dimshuffle(0, 2, 1)
                else:
                    baseline = baseline.dimshuffle(1, 0, 2)

        cost_std = TT.maximum(TT.sqrt(new_cost_var + 1e-8), 1.0)
        centered_reward = (reward - baseline - new_center) / cost_std

        if cost.ndim == 2:
            centered_reward = TT.addbroadcast(centered_reward, 1)

        N = probs.shape[-1]
        gradp = self.lambda1_reg * (centered_reward) * \
                (samples / (probsd + 1e-8)) + lambda2_reg * (TT.log(probsd + 1e-6) + as_floatX(1))

        if dimshuffle_probs:
            gradp = gradp.dimshuffle(0, 2, 1)

        if mask is not None:
            if self.generative_pred:
                gradp = mask.dimshuffle(0, 1, 'x') * gradp / N
            else:
                gradp = mask.dimshuffle(0, 1, 'x') * gradp

        known_grads = {probs: gradp}
        return updates, known_grads, new_center, cost_std, policy, lambda2_reg


class GumbelSoftmax(Operator):

    def __init__(self, itemp=1., use_log=False):
        super(GumbelSoftmax, self).__init__()
        self.itemp = itemp
        self.use_log = use_log

    def __call__(self, inp, trng):
        u_noise = trng.uniform(low=0., high=1., size=inp.shape)
        gumbel_noise = -TT.log(-TT.log(u_noise + 1e-5) + 1e-5)
        if self.use_log:
            gsoftmax = TT.nnet.softmax((TT.log(inp) + gumbel_noise) * (self.itemp))
        else:
            logits = (inp + gumbel_noise) * self.itemp

            # Clip the logits into a safe region for the float32
            # logits = logits.clip(-120, 80)
            # logits = theano.printing.Print("logits")(logits)
            gsoftmax = TT.nnet.softmax(logits)
        return gsoftmax


class GumbelSigmoid(Operator):

    def __init__(self, itemp=1., use_log=False):
        super(GumbelSigmoid, self).__init__()
        self.itemp = itemp
        self.use_log = use_log

    def __call__(self, inp, trng):
        u_noise = trng.uniform(low=0., high=1., size=inp.shape)
        gumbel_noise = -TT.log(-TT.log(u_noise + 1e-5) + 1e-5)
        if self.use_log:
            gsoftmax = HardSigmoid((TT.log(inp) + gumbel_noise) * (self.itemp))
            #gsoftmax = TT.nnet.sigmoid((TT.log(inp) + gumbel_noise) * (self.itemp))
        else:
            #gsoftmax = TT.nnet.sigmoid((inp + gumbel_noise) * (self.itemp))
            gsoftmax = HardSigmoid((inp + gumbel_noise) * self.itemp)
        return gsoftmax


class Dropout(Operator):

    def __init__(self, dropout_prob=0.5, rng=None):
        self.rng = RandomStreams(1) if rng is None else rng
        self.dropout_prob = dropout_prob

    """
        Perform the dropout on the layer.
    """
    def __call__(self, input, deterministic=False, dmask=None):

        if input is None:
            raise ValueError("input for the %s should not be empty." % __class__.__name__)

        p = self.dropout_prob
        if isinstance(deterministic, float) or isinstance(deterministic, bool):
            if deterministic:
                dropped_val = input
            else:
                retain_p = 1 - p
                input /= retain_p
                if dmask is None:
                    dropped_val = input * self.rng.binomial(input.shape,
                                                            p=retain_p,
                                                            dtype=floatX)
                else:
                    dropped_val = input * dmask
        else:
            retain_p = 1. - p
            if dmask is None:
                dropped_val = (1. - deterministic) * input + deterministic * (input / retain_p) * self.rng.binomial(input.shape,
                                                                                                                p=retain_p,
                                                                                                                dtype=floatX)
            else:
                dropped_val = (1. - deterministic) * input + deterministic * (input / retain_p) * dmask

        return dropped_val


class GaussianNoise(Operator):

    def __init__(self, avg=0, std=0.01, rng=None):
        self.rng = RandomStreams(1) if rng is None else rng
        self.avg = avg
        self.std = std

    def __call__(self):
        raise NotImplementedError("call function is not implemented!")


class AdditiveGaussianNoise(GaussianNoise):

    """
        Perform the dropout on the layer.
    """
    def __call__(self, input, deterministic=False):

       if input is None:
            raise ValueError("input for the %s should not be empty." % __class__.__name__)
       p = self.dropout_prob
       if deterministic:
           return input
       else:
           return input + self.rng.normal(input.shape,
                                          avg = self.avg,
                                          std = self.std,
                                          dtype=floatX)


class DiscreteOp(theano.gof.Op):
    __props__ = ("odtype",)

    def init(self, odtype="auto", axis=1):
        super(DiscreteOp, self).__init__()

        self.axis = axis
        self.odtype = odtype
        self.disc_out = None

        if axis > 1:
            return ValueError

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        try:
            self.odtype
        except AttributeError:
            self.odtype = 'auto'

    def make_node(self, x):
        x = TT.as_tensor_variable(x)
        out = TT.tensor(dtype=x.dtype, broadcastable=x.broadcastable)
        self.odtype = x.dtype
        return theano.Apply(self, [x], [out])

    def perform(self, node, inputs, outs):
        x = inputs[0]
        (z,) = outs
        if not hasattr(self, "axis"):
            self.axis = 1

        idxs = x.argmax(axis=self.axis)
        if z[0] is None or z[0].shape != x.shape[0]:
            z[0] = numpy.zeros(x.shape, dtype=node.outputs[0].dtype)

        if self.axis == 0:
            z[0][idxs, numpy.arange(x.shape[1])] = 1.
        else:
            z[0][numpy.arange(x.shape[0]), idxs] = 1.
        self.disc_out = z[0]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        #import pdb; pdb.set_trace()
        #z = TT.zeros_like(inputs[0]).astype("float32")
        #idxs = inputs[0].argmax(axis=1)
        #idxs2 = z.argmax(axis=1)
        #z = TT.set_subtensor(z[TT.arange(z.shape[0],), idxs], 1.)

        #z = TT.set_subtensor(z[TT.arange(z.shape[0],), idxs2], 1.)
        #z = theano.printing.Print("gradient mask: ")(z)



        return [output_grads[0]]

        #z = TT.zeros_like(inputs[0]).astype("float32")
        #idxs = z.argmax(axis=1)
        #z = TT.set_subtensor(z[TT.arange(z.shape[0], ), idxs], 1.)
        #return [z * output_grads[0]]

    def R_op(self, inputs, output_grads):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


concreteDist = DiscreteOp()


class DiscreteBernoulliOp(theano.gof.Op):
    __props__ = ("odtype",)

    def init(self, odtype="auto", axis=1):
        super(DiscreteOp, self).__init__()

        self.axis = axis
        self.odtype = odtype
        self.disc_out = None

        if axis > 1:
            return ValueError

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        try:
            self.odtype
        except AttributeError:
            self.odtype = 'auto'

    def make_node(self, x):
        x = TT.as_tensor_variable(x)
        out = TT.tensor(dtype=x.dtype, broadcastable=x.broadcastable)
        self.odtype = x.dtype
        return theano.Apply(self, [x], [out])

    def perform(self, node, inputs, outs):
        x = inputs[0]
        (z,) = outs
        if not hasattr(self, "axis"):
            self.axis = 1
        mask = (x > 0.5).astype(node.outputs[0].dtype)
        if z[0] is None or z[0].shape != x.shape[0]:
            z[0] = mask
        self.disc_out = z[0]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        z = (inputs[0] > 0.5).astype("float32")
        return [output_grads[0]]

    def R_op(self, inputs, output_grads):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


concreteBernDist = DiscreteBernoulliOp()


class MultiplicativeGaussianNoise(GaussianNoise):

    """
        Perform the dropout on the layer.
    """
    def __call__(self, input, deterministic=False):

       if input is None:
            raise ValueError("input for the %s should not be empty." % __class__.__name__)
       if deterministic:
           return input
       else:
           return input * self.rng.normal(input.shape,
                                          avg = self.avg,
                                          std = self.std,
                                          dtype=floatX)
       return result
