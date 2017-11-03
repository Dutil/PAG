import char_biscale
import char_biscale_planning
from matplotlib import pyplot as plt
import pickle as pkl
import mixer
import theano
import numpy
from collections import OrderedDict
import data_iterator
import nmt


def load_options(pkl_file):
    # load model model_options
    with open(pkl_file, 'rb') as f:
        options = pkl.load(f)

    return options

def get_one_batch(data_path_source, data_path_target, source_dic, target_dic, t_foo):

    valid = data_iterator.TextIterator(source=data_path_source,
                                       target=data_path_target,
                                       source_dict=source_dic,
                                       target_dict=target_dic,
                                       n_words_source=24440,
                                       n_words_target=302,
                                       source_word_level=1,
                                       target_word_level=0,
                                       batch_size=128,
                                       sort_size=20)

    alpha_value = None
    data = None
    for x, y in valid:
        # use_noise.set_value(1.)

        data = nmt.prepare_data(x, y, maxlen=50,
                                maxlen_trg=500,
                                n_words_src=24440,
                                n_words=302)

        alpha_value = t_foo(*data[:-1])

        break

    print "Done"
    return alpha_value, data

def compute_theano_foo(model_path, options, module):
    options['learn_t'] = False
    tparams_b = load_model(model_path, options, module, sampler=False)
    trng, use_noise, \
    x, x_mask, y, y_mask, \
    opt_ret, \
    cost = \
        module.build_model(tparams_b, options)

    inps = [x, x_mask, y, y_mask]
    alphas = opt_ret['dec_alphas']

    # For REINFORCE and stuff
    up = OrderedDict()
    if 'dec_updates' in opt_ret:
        up = opt_ret['dec_updates']
        commits = opt_ret['dec_commits']
        f_alpha = theano.function(inps, [alphas,commits], profile=False, updates=up)

    else:
        f_alpha = theano.function(inps, alphas, profile=False, updates=up)


    return f_alpha


def load_dicts(dictionary, dictionary_target):
    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
        word_idict[kk] = vv
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    return word_idict, word_idict_trg


def load_data(source_file, source_dic, options, nb=None):
    data = []

    with open(source_file, 'r') as f:

        for idx, line in enumerate(f):

            words = line.strip().split()
            x = map(lambda w: source_dic[w] if w in source_dic else 1, words)
            x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
            x += [0]

            data.append(x)

            if nb is not None and idx >= nb:
                break

    return data


def load_model(model_path, options, module, sampler=True):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = module.init_params(options)

    # load model parameters and set theano shared variables
    params = mixer.load_params(model_path, params)
    tparams = mixer.init_tparams(params)
    use_noise = theano.shared(numpy.float32(0.))

    if sampler:
        f_init, f_next = module.build_sampler(tparams, options, trng, use_noise)
        return tparams, f_init, f_next
    else:
        return tparams


def translate(tparams, f_init, f_next, options, source, module, k=5, normalize=False):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # word index
    use_noise = theano.shared(numpy.float32(0.))

    # f_init, f_next = module.build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        use_noise.set_value(0.)
        # sample given an input sequence and obtain scores
        sample, score, alphas, commits = module.gen_sample(tparams, f_init, f_next,
                                                           numpy.array(seq).reshape([len(seq), 1]),
                                                  options, trng=trng, k=k, maxlen=500,
                                                  stochastic=False, argmax=False, send_alpha=True)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx], alphas[sidx], commits[sidx]

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

    return translation, alphas_sample, commits_sample


def convert(caps, dic, char_lvl=True):
    # utility function
    capsw = []
    for cc in caps:
        ww = []
        for w in cc:
            if w == 0:
                break
            try:
                ww.append(dic[w].encode('utf-8', 'ignore'))
            except:
                ww.append('*'.encode("utf-8"))
        if char_lvl:
            capsw.append(''.join(ww))
        else:
            capsw.append(' '.join(ww))

    return capsw

def show_alpha(alphas, normalize=False, commits=None):

    if normalize:
        alphas = np.exp(alphas) / np.sum(np.exp(alphas), axis=0)

    plt.imshow(alphas)
    cb = plt.colorbar()
    plt.show()

    if commits is not None:
        plt.imshow(alphas + commit)
        cb = plt.colorbar()
        plt.show()

        plt.imshow(np.zeros_like(alphas) + commit)
        cb = plt.colorbar()
        plt.show()