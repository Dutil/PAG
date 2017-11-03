'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import cPickle as pkl

from da2qgen import gen_sample
from model_utils import init_tparams, load_params
from model import Model

from multiprocessing import Process, Queue
import os

import nltk


class BeamSearch(object):
    """
        Perform the Constrained Beam Search.
    """
    def __init__(self, nprocs,
                 beam_size,
                 constraints=None,
                 normalize=False,
                 data_iter=None,
                 lower_case=False,
                 n_processes=5,
                 max_doc_len=50,
                 max_sample_len=200):

        self.nprocs = nprocs
        self.beam_size = beam_size
        self.n_processes = n_processes
        self.constraints = constraints
        self.normalize = normalize
        self.lower_case = lower_case
        self.data_iter = data_iter
        self.max_sample_len = max_sample_len
        self.max_doc_len = max_doc_len

    def gen_model(self, queue=None, rqueue=None, pid=0, mpath=None, options=None):
        """
            Load the model and start getting the information from the queue.
        """
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

        if mpath is None:
            raise ValueError("mpath argument to the gen_model function should not be empty")

        if queue is None:
            raise ValueError("queue should not be empty.")

        if rqueue is None:
            raise ValueError("rqueue should not be empty.")

        trng = RandomStreams(1234)
        model = Model(prefix="da2qmodel",
                      options=options)

        # allocate model parameters
        params = model.init_params(options)

        # load model parameters and set theano shared variables
        params = load_params(mpath, params)
        tparams = init_tparams(params)

        # word index
        f_init, f_next = model.build_sampler(tparams, options, trng)
        def _gen_question(doc, doc_mask, ans, ans_mask):
            doc = numpy.asarray(doc, dtype="int64")
            doc_mask = numpy.asarray(doc_mask, dtype="float32")
            ans = numpy.asarray(ans, dtype="int64")
            ans_mask = numpy.asarray(ans_mask, dtype="float32")

            # sample given an input sequence and obtain scores
            sample, score = gen_sample(tparams,
                                       f_init,
                                       f_next,
                                       doc=doc[:, None],
                                       ans=ans[:, None],
                                       doc_mask=doc_mask[:, None],
                                       ans_mask=ans_mask[:, None],
                                       options=options,
                                       trng=trng,
                                       k=self.beam_size,
                                       maxlen=self.max_sample_len,
                                       stochastic=False,
                                       argmax=False)

            # normalize scores according to sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in sample])
                score = score / lengths

            sidx = numpy.argmin(score)
            return sample[sidx]

        while True:
            req = queue.get()
            if req is None:
                break
            idx, bvals = req[0], req[1]

            doc = bvals[0]
            doc_mask = bvals[1]
            question = bvals[2]
            question_mask = bvals[3]
            ans = bvals[4]
            ans_mask = bvals[5]

            print pid, '-', idx

            seq = _gen_question(doc, doc_mask, ans, ans_mask)
            rqueue.put((idx, (seq, doc)))
        return

    def perform(self,
                model,
                dicts_file,
                saveto,
                chr_level=False):
        """
            Multi-threaded beam-search code. This is the main part
            of the code where the most of the functions about
            the model is being called.
        """

        # load model model_options
        with open('%s.pkl' % model, 'rb') as f:
            options = pkl.load(f)

        # load source dictionary and invert
        with open(dicts_file, 'rb') as f:
            dicts = pkl.load(f)
            doc_dict = dicts[0]
            q_dict = dicts[1]

        doc_idict = dict()
        for kk, vv in doc_dict.iteritems():
            doc_idict[vv] = kk

        q_idict = dict()
        for kk, vv in q_dict.iteritems():
            q_idict[vv] = kk

        q_idict_set = set(q_idict.keys())
        # create input and output queues for processes
        queue = Queue()
        rqueue = Queue()
        processes = [None] * self.n_processes

        # Create the processes to generate
        # the translations
        for midx in xrange(self.n_processes):
            processes[midx] = Process(
                target=self.gen_model,
                args=(queue, rqueue, midx, model, options))
            processes[midx].start()

        # utility function
        def _seqs2words(seqs):
            genseqs = []

            for i, (cc, all_xcc) in enumerate(seqs):
                ww = []
                for j, w in enumerate(cc):
                    if numpy.array(cc)[j:].sum() == 0 or cc[j] == q_dict['EOS']:
                        break

                    maxlen = self.max_doc_len

                    if w >= maxlen:
                        widx = w - maxlen
                        if widx == 0:
                            break
                        if w in q_idict:
                            ww.append(q_idict[widx])
                        else:
                            ww.append("UNK")
                    elif w == maxlen - 1:
                        ww.append("UNK")
                    else:
                        sw = all_xcc[w]
                        if sw in doc_dict:
                            ww.append(sw)
                        else:
                            ww.append(sw)
                genseqs.append(' '.join(ww))

            return genseqs

        def _send_jobs(data_iter):
            i = 0
            for idx, sample in enumerate(data_iter):
                doc, doc_mask, question, question_mask, ans, ans_mask = sample[0], sample[1], \
                        sample[2], sample[3], sample[4], sample[5]
                queue.put((idx, (doc, doc_mask, question, question_mask, ans, ans_mask)))

            return idx + 1

        def _finish_processes():
            for midx in xrange(self.n_processes):
                queue.put(None)

        def _retrieve_jobs(n_samples):
            trans = [None] * n_samples
            for idx in xrange(n_samples):
                resp = rqueue.get()
                trans[resp[0]] = resp[1]
                if numpy.mod(idx, 10) == 0:
                    print 'Sample ', (idx + 1), '/', n_samples, ' Done'
            return trans

        print 'Generating questions...'
        n_samples = _send_jobs(self.data_iter)

        qs = _seqs2words(_retrieve_jobs(n_samples))
        _finish_processes()

        with open(saveto, 'w') as f:
            print >>f, '\n'.join(qs)
        print 'Done'


if __name__ == "__main__":

    fpath = "/data/lisatmp4/gulcehrc/nmt/data/"
    #
    cwords_fil = os.path.join(fpath, common_wordsd)
    words_trdf = os.path.join(fpath, words_trd)

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=6)
    parser.add_argument('-p', type=int, default=7)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('-fd', default=cwords_fil)
    parser.add_argument('-wd', default=words_trdf)

    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()
    beamsearch = BeamSearch(n_processes=args.p,
                            beam_size=args.k,
                            normalize=args.n,
                            data_iter=bdata_iter,
                            lower_case=True)

    """
    main(args.model,
         args.dictionary,
         args.dictionary_target, args.source,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p,
         chr_level=args.c, commons_map=cwords_fil,
         fren_word_map=words_trdf)
    """
