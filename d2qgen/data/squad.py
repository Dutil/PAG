#from __future__ import print_function

from six import Iterator
import json
import numpy as np
#import d2qgen
from data import DataMode
from collections import OrderedDict, Counter
import nltk


class SQUADIterator(Iterator):
    """
        A stateful data iterator.
    """
    def __init__(self,
                 path=None,
                 mode=DataMode.train,
                 rng=None,
                 nwords_doc=None,
                 nwords_q=None,
                 use_pointers=False,
                 nwords_ans=None,
                 batch_size=100,
                 use_doc_emb_ans=False,
                 use_lowercase=True,
                 dicts=None,
                 assign_priority_point=False,
                 use_char_rep=False,
                 maxlen=None,
                 val_size=-1,
                 val_idxs=None,
                 buffer_size=200):

        self.path = path
        self.mode = mode
        self.val_size = val_size

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.use_lowercase = use_lowercase
        self.use_pointers = use_pointers
        self.maxlen = maxlen
        self.use_char_rep = use_char_rep
        self.use_doc_emb_ans = use_doc_emb_ans

        # Use as much as pointers you can use to the source the rest will be
        # completed from shortlist vocabulary.
        self.assign_priority_point = assign_priority_point

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(1)

        self.data = OrderedDict({})
        self.data_len = 0
        self.nwords_doc = nwords_doc
        self.nwords_q = nwords_q
        self.nwords_ans = nwords_ans

        self.dicts = dicts
        self.wFreq = None
        self.qwFreq = None
        self.offset = 0

        self.doc_lens = []
        self.q_lens = []
        self.ans_lens = []
        self.queue = []

        if self.nwords_doc is not None:
            if self.nwords_doc < 0:
                raise ValueError("Number of words should be greater than 0.")

        self.q2doc_dict = OrderedDict({})

        self.idxs = None
        self.val_idxs = val_idxs

        self.eidx = 0
        self.doc_dict = OrderedDict({})
        self.q_dict = OrderedDict({})
        self.ans_dict = OrderedDict({})

        self.__load_data()
        self.__count_tokens()

    def __load_data(self):

        print "loading the files."
        with open(self.path) as dpath:
            json_content = json.load(dpath)
            self.doc_data = json_content['data']

    def __populate_commons_dict(self):
        doc_keys = set(self.doc_dict.keys())
        for i, (w, idx) in enumerate(self.q_dict.iteritems()):
            if w in doc_keys:
                self.q2doc_dict[idx] = self.doc_dict[w]

        del self.q2doc_dict[0]
        del self.q2doc_dict[1]
        self.common_words_dict = OrderedDict({v:k for k, v in self.q2doc_dict.iteritems()})

    def __count_tokens(self):
        print "Creating dictionaries and parsing doc."
        words = []
        q_words = []
        ans_words = []
        maxlen = 0
        for doc in self.doc_data:
            doc = doc['paragraphs']
            for el in doc:
                context = el['context']
                qas = el['qas']
                if self.use_lowercase:
                    context = context.lower()

                wordsInContext = nltk.word_tokenize(context)
                words.extend(wordsInContext)

                for qa in qas:
                    wordsInQuestion = qa["question"]
                    anss = qa['answers']

                    wordsInQuestion = wordsInQuestion.lower() \
                            if self.use_lowercase else wordsinquestion

                    ans_lens = []
                    ans_start = 0

                    if self.mode == DataMode.train:
                        ans = anss[np.random.random_integers(low=0, high=len(anss)) - 1]
                    else:
                        ans = anss[0]

                    remaining = len(nltk.word_tokenize(context[ans["answer_start"]:]))
                    start_idx = len(wordsInContext) - remaining
                    ans_len = len(nltk.word_tokenize(ans['text']))

                    if start_idx > self.maxlen - ans_len:
                        ans = anss[0]
                        if start_idx >= self.maxlen - ans_len:
                            start_idx = self.maxlen - ans_len - 1

                    wordsInAnswer = ans['text'].lower() if self.use_lowercase else ans['text']
                    ans_start = int(start_idx)

                    if not self.use_char_rep:
                        wordsInAnswer = nltk.word_tokenize(wordsInAnswer)[:len(wordsInAnswer) + 1]
                        ans_locs = list(ans_start + np.arange(0, ans_len))
                        ans_words.extend(wordsInAnswer)
                        ans_lens.append(len(wordsInAnswer))
                    else:
                        charsInAnswer = list(wordsInAnswer)
                        ans_locs = list(ans_start + np.arange(0, ans_len))[:len(charsInAnswer) + 1]
                        ans_words.extend(wordsInAnswer)
                        ans_lens.append(len(charsInAnswer))

                    wordsInQuestion = nltk.word_tokenize(wordsInQuestion)
                    q_words.extend(wordsInQuestion)

                    if self.maxlen is None or self.maxlen <= 0:
                        if len(wordsInContext) + 2 >= maxlen:
                            maxlen = len(wordsInContext) + 2

                    self.doc_lens.append(len(wordsInContext) + 1)
                    self.q_lens.append(len(wordsInQuestion) + 1)
                    self.ans_lens.append(ans_lens[0] + 1)

                    self.data[qa["question"]] = OrderedDict({})
                    self.data[qa["question"]]["text"] = context
                    self.data[qa["question"]]["answer"] = ans['text']
                    self.data[qa["question"]]["ans_locs"] = ans_locs
                    self.data[qa['question']]['answer_start'] = ans_start

        sorted_idxs = np.argsort(self.doc_lens)
        if self.maxlen is None or self.maxlen <= 0:
            self.maxlen = maxlen

        if not self.dicts:
            self.wFreq = Counter(words)
            self.wFreq = self.wFreq.most_common()

            self.qwFreq = Counter(q_words)
            self.qwFreq = self.qwFreq.most_common()

            self.awFreq = Counter(ans_words)
            self.awFreq = self.awFreq.most_common()

            print "The number of words in the documents.", len(self.wFreq)
            print "The number of words in the questions.", len(self.qwFreq)
            print "The number of words in the answers.", len(self.awFreq)

            print "The maximum document length is ", np.max(self.doc_lens)
            print "The maximum answer length is ", np.max(self.ans_lens)
            print "The maximum question length is ",  np.max(self.q_lens)

            self.doc_dict['UNK'], self.q_dict['UNK'], self.ans_dict['UNK'] = 1, 1, 1
            self.doc_dict['EOS'], self.q_dict['EOS'], self.ans_dict['EOS'] = 0, 0, 0

            for i, el in enumerate(self.wFreq):
                self.doc_dict[el[0]] = i + 2

            for i, el in enumerate(self.qwFreq):
                self.q_dict[el[0]] = i + 2

            for i, el in enumerate(self.awFreq):
                self.ans_dict[el[0]] = i + 2
        else:
            self.doc_dict = self.dicts[0]
            self.q_dict = self.dicts[1]
            self.ans_dict = self.dicts[2]

        self.data_len = len(self.data)
        self.idxs = list(range(self.data_len))

        if self.val_idxs:
            self.idxs = self.val_idxs
        else:
            if self.val_size > 0:
                self.val_idxs = self.idxs[:self.val_size]
                self.idxs = self.idxs[self.val_size:]

        if self.nwords_doc is None or self.nwords_doc <= 0:
            self.nwords_doc, self.nwords_q, self.nwords_ans = len(self.doc_dict), \
                    len(self.q_dict), len(self.ans_dict)

        self.data = list(self.data.items())
        self.idoc_dict = {v:k for k, v in self.doc_dict.iteritems()}
        if self.use_pointers:
            print "populating the commons q to doc dictionary."
            self.__populate_commons_dict()

    def __encode_int_rep(self, doc, q, ans):
        encoded_q = []
        encoded_doc = []
        encoded_ans = []
        encoded_allq = []
        encoded_alldoc = []

        for w in doc:
            if w in self.doc_dict:
                encoded_alldoc.append(self.doc_dict[w])
                if self.doc_dict[w] < self.nwords_doc:
                    encoded_doc.append(self.doc_dict[w])
                else:
                    encoded_doc.append(self.doc_dict['UNK'])
            else:
                encoded_doc.append(self.doc_dict['UNK'])
                encoded_alldoc.append(self.doc_dict['UNK'])

        encoded_alldoc.append(self.doc_dict['EOS'])
        encoded_doc.append(self.doc_dict['EOS'])

        for w in q:
            if w in self.q_dict:
                encoded_allq.append(self.q_dict[w])
                if self.q_dict[w] < self.nwords_q:
                    encoded_q.append(self.q_dict[w])
                else:
                    encoded_q.append(self.q_dict['UNK'])
            else:
                encoded_q.append(self.q_dict['UNK'])
                encoded_allq.append(self.q_dict['UNK'])

        encoded_allq.append(self.q_dict['EOS'])
        encoded_q.append(self.q_dict['EOS'])

        for w in ans:
            if w in self.ans_dict:
                if self.ans_dict[w] < self.nwords_ans:
                    encoded_ans.append(self.ans_dict[w])
                else:
                    encoded_ans.append(self.ans_dict['UNK'])
            else:
                encoded_ans.append(self.ans_dict['UNK'])
        encoded_ans.append(self.ans_dict['EOS'])

        return encoded_doc, encoded_q, encoded_ans, encoded_allq, encoded_alldoc

    def reset(self):
        self.offset = 0
        self.queue = []
        self.eidx = 0

    def __iter__(self):
        return self

    def __search_word(self, w, ans_start, doc):

        # Return the word that is closest to the answer
        locs = []
        nchars = 0

        if w > 1:
            # find the locations of the word in the source doc.
            for i, d in enumerate(doc):
                if i >= self.maxlen:
                    break

                nchars += len(self.idoc_dict[d]) + 1

                if w in self.q2doc_dict and self.q2doc_dict[w] == d:
                    locs.append((i, nchars))

        # find the closest word to the answer
        if locs:
            dists = [abs(ans_start - l[1]) for l in locs]
            return locs[np.argmin(dists)][0]
        else:
            return self.maxlen + 1

    def next(self):
        if len(self.queue) == 0:
            if self.eidx >= 1:
                raise StopIteration

            if self.mode == DataMode.train:
                np.random.shuffle(self.idxs)

            self.eidx += 1
            self.queue.extend(self.idxs)

        batch_idxs, self.queue = self.queue[-self.batch_size:], self.queue[:-self.batch_size]
        lens, qlens, ans_lens = [], [], []
        docs, qs, anss, ans_locs = [], [], [], []
        all_qs = []
        all_docs = []
        ans_starts = []

        for bidx in batch_idxs:

            if self.use_lowercase:
                doc = self.data[bidx][1]['text'].lower()
                q = self.data[bidx][0].lower()
                ans = self.data[bidx][1]['answer'].lower()
            else:
                doc = self.data[bidx][1]['text']
                q = self.data[bidx][0]
                ans = self.data[bidx][1]['answer']

            ans_starts.append(self.data[bidx][1]['answer_start'])
            ans_locs.append(self.data[bidx][1]['ans_locs'])

            doc = nltk.word_tokenize(doc)
            q = nltk.word_tokenize(q)
            ans = nltk.word_tokenize(ans)

            enc_doc, enc_q, enc_ans, encoded_allq, encoded_alldoc = self.__encode_int_rep(doc, q, ans)
            all_qs.append(encoded_allq)
            all_docs.append(encoded_alldoc)

            docs.append(enc_doc)
            qs.append(enc_q)
            anss.append(enc_ans)

            lens.append(len(enc_doc))
            qlens.append(len(enc_q))
            ans_lens.append(len(enc_ans))

        max_doc_len = self.maxlen

        if self.use_pointers:
            for j, que in enumerate(qs):
                for i, w in enumerate(que):
                    awq = all_qs[j][i]
                    ad = all_docs[j]
                    if self.assign_priority_point:
                            new_w = self.__search_word(awq, ans_starts[j], ad)
                            if new_w < self.maxlen:
                                qs[j][i] = new_w
                            else:
                                qs[j][i] += max_doc_len
                    else:
                        if w == 1 and awq in self.q2doc_dict:
                            new_w = self.__search_word(awq, ans_starts[j], ad)
                            qs[j][i] = new_w
                        else:
                            qs[j][i] += max_doc_len

        doc_mask = np.zeros((max_doc_len, len(lens))).astype("float32")
        q_mask = np.zeros((np.max(qlens), len(lens))).astype("float32")
        ans_mask = np.zeros((np.max(ans_lens), len(lens))).astype("float32")

        for i, batch_idx in enumerate(batch_idxs):
            doc_mask[:lens[i], i] = 1.
            q_mask[:qlens[i], i] = 1.
            ans_mask[:ans_lens[i], i] = 1.

        return docs, doc_mask, qs, q_mask, anss, ans_locs, ans_mask

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['data']
        return state

    def __setstate__(self):
        self.__dict__.update(d)
        self.__load_data()
        self.__count_tokens()


if __name__=="__main__":
    import os
    path = "/home/caglar.gulcehre/data/squad/"
    ddir_ = os.path.join(path, "dev-v1.1.json")
    tdir_ = os.path.join(path, "train-v1.1.json")

    squad = SQUADIterator(path=tdir_,
                          nwords_doc=60000,
                          nwords_ans=1000,
                          nwords_q=1000,
                          maxlen=100)

    squadv = SQUADIterator(path=ddir_,
                           mode=DataMode.valid,
                           use_pointers=True,
                           nwords_doc=60000,
                           nwords_ans=1000,
                           nwords_q=1000,
                           maxlen=100,
                           dicts=[squad.doc_dict,
                                  squad.q_dict,
                                  squad.ans_dict])

    import pdb; pdb.set_trace()
    dec_doc, doc_mask, dec_q, q_mask, anss, _, ans_mask = next(squadv)
