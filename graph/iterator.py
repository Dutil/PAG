import numpy
import os
import random

import cPickle
import gzip
import codecs


numpy.random.seed(12456)


class GraphIterator(object):

    def __init__(self,
                 source, source_dict,
                 batch_size=128,
                 job_id=0,
                 sort_size=20,
                 shuffle_per_epoch=True):

        self.source_file = source

        data = cPickle.load(open(source))
        self.source = data['inputs']
        self.source_dict = cPickle.load(open(source_dict))
        self.target = data['targets']

        self.batch_size = batch_size
        self.shuffle_per_epoch = shuffle_per_epoch

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * sort_size

        self.end_of_data = False
        self.job_id = job_id

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle_per_epoch:
            # close current files
            self.shuffle()

    def shuffle(self):

        idx = range(len(self.source))
        numpy.random.shuffle(idx)
        self.source = [self.source[i] for i in idx]
        self.target = [self.target[i] for i in idx]

    def next(self):

        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        source_lens = []
        target_lens = []

        # fill buffer, if it's empty
        if self.target is not None:
            assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k, [input, tar] in enumerate(zip(self.source, self.target)):

                self.source_buffer.append(input)
                self.target_buffer.append(tar)

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    self.end_of_data = True
                    break

                source.append(ss)
                source_lens.append(len(ss))

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                #adding the final 0
                #tt.append(0)
                target.append(tt + [0])
                target_lens.append(len(tt) + 1)

                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True


        source_mask = numpy.zeros((numpy.max(source_lens), len(source_lens))).astype("float32")
        target_mask = numpy.zeros((numpy.max(target_lens), len(target_lens))).astype("float32")

        for i, batch_idx in enumerate(source):
            source_mask[:source_lens[i], i] = 1.
            target_mask[:source_lens[i], i] = 1.

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        return source, target



if __name__=="__main__":
    import os
    path = "/data/lisatmp4/gulcehrc/data/graphprobs/noisy_rnn/train/"
    train = "4_rnn.txt.pkl"
    valid = "4_rnn.txt.val.pkl"
    source_dict= "4_rnn.txt.dict.pkl"

    train = GraphIterator(source=path+train,
                          source_dict=path+source_dict)

    train = GraphIterator(source=path+valid,
                          source_dict=path+source_dict)

    #import pdb; pdb.set_trace()
    source, source_mask, target, target_mask = train.next()
    #print source, source_mask, target, target_mask
