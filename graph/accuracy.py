from matplotlib import pyplot as plt
import pickle as pkl
import mixer
import theano
import numpy
from collections import OrderedDict



def translate(tparams, f_init, f_next, options, source, gen_sample, k=5, normalize=False):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # word index
    use_noise = theano.shared(numpy.float32(0.))

    # f_init, f_next = module.build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        use_noise.set_value(0.)
        # sample given an input sequence and obtain scores
        sample, score, alphas, commits = gen_sample(tparams, f_init, f_next,
                                                    numpy.array(seq).reshape([len(seq), 1]),
                                                    options, trng=trng, k=k, maxlen=10,
                                                    stochastic=True, argmax=True, send_alpha=True)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        #sidx = numpy.argmin(score)
        #areturn sample[sidx], alphas[sidx], commits[sidx]
        return sample, alphas, commits

    translation = []
    alphas_sample = []
    commits_sample = []
    for req in source:

        if req is None:
            break

        #seq, alphas = _translate(req)
        seq, alphas, commits = _translate(req)
        translation.append(seq)
        alphas_sample.append(alphas)
        commits_sample.append(commits)

    return translation
