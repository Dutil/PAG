import sys
import logging
import numpy as np
import cPickle as pkl
import yaml
import os
from keras.preprocessing.sequence import pad_sequences
import logging

from mrclib.dataset.base import NEWSQADataset
from mrclib.dataset.base import shuffle_data_dict


logger = logging.getLogger(__name__)

TRAIN_SIZE = -1
VALID_SIZE = -1
TEST_SIZE = -1
batch_size_train = 32
batch_size_eval = 32
id2w = None
write_counter = 0


def trim_batch(batch, trim_margin=None):
    #  batch.shape: N * n_words
    if trim_margin is None:
        non_zero = batch > 0
        while non_zero.ndim > 2:
            non_zero = np.max(non_zero, axis=-1)
        nz_index = np.argmax(non_zero, axis=1)
        trim_margin = np.min(nz_index)
    return batch[:, trim_margin:], trim_margin


def multiword_trim(batch_dict):

    batch_dict['input_story'], story_margin = trim_batch(batch_dict['input_story'])
    batch_dict['boundary'], _ = trim_batch(batch_dict['boundary'], story_margin)
    batch_dict['input_question'], _ = trim_batch(batch_dict['input_question'])
    return batch_dict


def simple_generator(data_dict, input_keys, output_keys,
                     batch_size, trim_function=None):
    sample_count = None
    for k, v in data_dict.items():
        if sample_count is None:
            sample_count = v.shape[0]
        if not (sample_count == v.shape[0]):
            raise Exception('Mismatched sample counts in data_dict.')

    batches_per_epoch = sample_count // batch_size
    if sample_count % batch_size > 0:
        batches_per_epoch += 1

    while True:
        for batch_num in range(batches_per_epoch):
            # grab the chunk of samples in the current bucket
            batch_start = batch_num * batch_size
            batch_end = batch_start + batch_size
            if batch_start >= sample_count:
                continue
            batch_end = min(batch_end, sample_count)
            batch_idx = np.arange(batch_start, batch_end)
            batch_dict = {k: v.take(batch_idx, axis=0) for k, v in data_dict.iteritems()}
            if trim_function is not None:
                batch_dict = trim_function(batch_dict)

            batch_x = [batch_dict[k] for k in input_keys]
            batch_y = [batch_dict[k] for k in output_keys]
            yield batch_x, batch_y


def random_generator(data_dict, input_keys, output_keys, batch_size,
                     bucket_size=1000, sort_by=None, trim_function=None):
    sample_count = None
    for k, v in data_dict.items():
        if sample_count is None:
            sample_count = v.shape[0]
        if not (sample_count == v.shape[0]):
            raise Exception('Mismatched sample counts in data_dict.')

    if bucket_size > sample_count:
        bucket_size = sample_count
        logger.warn('bucket_size < sample_count')
    # epochs discard dangling samples that won't fill a bucket.
    buckets_per_epoch = sample_count // bucket_size
    if sample_count % bucket_size > 0:
        buckets_per_epoch += 1

    while True:
        # random shuffle
        data_dict = shuffle_data_dict(data_dict)

        for bucket_num in range(buckets_per_epoch):
            # grab the chunk of samples in the current bucket
            bucket_start = bucket_num * bucket_size
            bucket_end = bucket_start + bucket_size
            if bucket_start >= sample_count:
                continue
            bucket_end = min(bucket_end, sample_count)
            current_bucket_size = bucket_end - bucket_start
            bucket_idx = np.arange(bucket_start, bucket_end)
            bucket_dict = {k: v.take(bucket_idx, axis=0) for k, v in data_dict.iteritems()}

            if sort_by is not None:
                non_zero = bucket_dict[sort_by]
                while non_zero.ndim > 2:
                    non_zero = np.max(non_zero, axis=-1)
                pad_counts = np.sum((non_zero == 0), axis=1)
                sort_idx = np.argsort(pad_counts)
                bucket_dict = {k: v.take(sort_idx, axis=0) for k, v in bucket_dict.iteritems()}

            batches_per_bucket = current_bucket_size // batch_size
            if current_bucket_size % batch_size > 0:
                batches_per_bucket += 1

            for batch_num in range(batches_per_bucket):
                # grab the chunk of samples in the current bucket
                batch_start = batch_num * batch_size
                batch_end = batch_start + batch_size
                if batch_start >= current_bucket_size:
                    continue
                batch_end = min(batch_end, current_bucket_size)
                batch_idx = np.arange(batch_start, batch_end)
                batch_dict = {k: v.take(batch_idx, axis=0) for k, v in bucket_dict.iteritems()}

                if trim_function is not None:
                    batch_dict = trim_function(batch_dict)

                batch_x = [batch_dict[k] for k in input_keys]
                batch_y = [batch_dict[k] for k in output_keys]
                yield batch_x, batch_y


############################################
# training
############################################
def filter_keys(train, valid, test):
    new_data = []

    def _in_range(_range_list, _id):
        for _r in _range_list:
            h, t = _r.split(':', 1)
            h, t = int(h), int(t)
            if _id >= h and _id <= t:
                return True
        return False

    for ds in [train, valid, test]:
        r = dict()
        pop_answer_range_list = ds['popular_answer_ranges'][0].split('\n')

        question_list = []
        story_list = []
        answer_range_list = []
        answer_list = []

        for i, _range in enumerate(pop_answer_range_list):
            if i >= len(ds['input_story']) or i >= len(ds['input_question']):
                break
            if _range == '':
                continue
            story = np.trim_zeros(ds['input_story'][i])
            answer_ranges, answers = [], []
            _range_list = _range.split(',')

            _range_list = [_r for _r in _range_list if r is not '']
            if len(_range_list) == 0:
                continue
            head = int(_range_list[0].split(':', 1)[0])
            tail = int(_range_list[-1].split(':', 1)[1])
            if tail <= head:
                continue
            answer_ranges.append((head, tail - 1))
            _a = []
            for _r in _range_list:
                head = int(_r.split(':', 1)[0])
                tail = int(_r.split(':', 1)[1])
                _a.append(story[head: tail])
            _a = np.concatenate(_a, 0)
            answers.append(_a)

            answer_range_list.append(answer_ranges)
            answer_list.append(answers)
            story_list.append(story)
            question_list.append(ds['input_question'][i])

        r['input_story'] = story_list
        r['input_question'] = question_list
        r['input_answer'] = answer_list
        r['boundary'] = answer_range_list
        new_data.append(r)

    max_story_word_amount = max(
        map(len, new_data[0]['input_story'] + new_data[1]['input_story'] + new_data[2]['input_story']))
    max_question_word_amount = max(
        map(len, new_data[0]['input_question'] + new_data[1]['input_question'] + new_data[2]['input_question']))
    max_answer_amount = max(
        map(len, new_data[0]['input_answer'] + new_data[1]['input_answer'] + new_data[2]['input_answer']))
    max_answer_word_amount = max(
        map(len, [word for answer in new_data[0]['input_answer'] +
                  new_data[1]['input_answer'] +
                  new_data[2]['input_answer'] for word in answer]))

    story_len = [[], [], []]
    for idx in xrange(3):
        for s in new_data[idx]['input_story']:
            story_len[idx].append(len(s))

    for idx in xrange(3):

        new_data[idx]['input_story'] = pad_sequences(new_data[idx]['input_story'], maxlen=max_story_word_amount).astype('int32')
        new_data[idx]['input_question'] = pad_sequences(new_data[idx]['input_question'], maxlen=max_question_word_amount).astype('int32')

    counter = 0
    for idx in xrange(3):
        answers = new_data[idx]['input_answer']
        ranges = new_data[idx]['boundary']
        stories = new_data[idx]['input_story']
        pad_ans = np.zeros([len(answers), max_answer_amount, max_answer_word_amount], dtype='int32')
        for i in range(len(answers)):
            nb_a = len(answers[i])
            for j, k in zip(range(0, nb_a), range(-nb_a, 0)):
                nb_w = len(answers[i][j])
                pad_ans[i, k, -nb_w:] = answers[i][j]

        pad_range = np.zeros(stories.shape + (2,), dtype='int32')
        for i in range(len(answers)):
            if max_story_word_amount - story_len[idx][i] + ranges[i][0][0] >= max_story_word_amount or\
                    max_story_word_amount - story_len[idx][i] + ranges[i][0][1] >= max_story_word_amount:
                print(idx, i)
                counter += 1
            if max_story_word_amount - story_len[idx][i] + ranges[i][0][0] >= max_story_word_amount:
                head = max_story_word_amount - 1
            else:
                head = max_story_word_amount - story_len[idx][i] + ranges[i][0][0]
            if max_story_word_amount - story_len[idx][i] + ranges[i][0][1] >= max_story_word_amount:
                tail = max_story_word_amount - 1
            else:
                tail = max_story_word_amount - story_len[idx][i] + ranges[i][0][1]

            pad_range[i][head][0] = 1
            pad_range[i][tail][1] = 1

        new_data[idx]['input_answer'] = pad_ans
        new_data[idx]['boundary'] = pad_range

    return new_data[0], new_data[1], new_data[2]


def add_shape_to_metadata(metadata, dataset):
    for k, v in dataset.iteritems():
        shape_key = k + "_shape"
        metadata[shape_key] = dataset[k].shape


def recover_real_split_size(dataset, key='input_question'):
    global TRAIN_SIZE, VALID_SIZE, TEST_SIZE
    TRAIN_SIZE = dataset[0][key].shape[0]
    VALID_SIZE = dataset[1][key].shape[0]
    TEST_SIZE = dataset[2][key].shape[0]


def train_dataset(config_dir='config', update_dict=None):
    if not model_config['dataset']['h5'].startswith('newsqa'):
        print('need to be newsqa data... check your config file...')
        return

    dataset = NEWSQADataset(dataset_h5=model_config['dataset']['h5'], data_path='/home/eric.yuan/newsqa_dataset_9k_6/')
    train_data, valid_data, test_data = dataset.get_data(train_size=TRAIN_SIZE, valid_size=VALID_SIZE,
                                                         test_size=TEST_SIZE)

    train_data, valid_data, test_data = filter_keys(train_data, valid_data, test_data)
    train_data['input_question'] = np.concatenate([train_data['input_question'], np.ones([train_data['input_question'].shape[0], 1], dtype='int32')], axis=1)
    valid_data['input_question'] = np.concatenate([valid_data['input_question'], np.ones([valid_data['input_question'].shape[0], 1], dtype='int32')], axis=1)
    test_data['input_question'] = np.concatenate([test_data['input_question'], np.ones([test_data['input_question'].shape[0], 1], dtype='int32')], axis=1)

    recover_real_split_size((train_data, valid_data, test_data))

    add_shape_to_metadata(dataset.meta_data, train_data)

    global id2w
    id2w = dataset.meta_data['id2word']
    for k, v in dataset.meta_data.iteritems():
        if k.endswith('_shape'):
            print(k, v)

    input_keys = ['input_story', 'input_question']
    output_keys = ['boundary']

    train_batch_generator = random_generator(data_dict=train_data, batch_size=batch_size_train,
                                             input_keys=input_keys, output_keys=output_keys,
                                             trim_function=multiword_trim, sort_by='input_story')

    valid_batch_generator = random_generator(data_dict=valid_data, batch_size=batch_size_eval,
                                             input_keys=input_keys, output_keys=output_keys,
                                             trim_function=multiword_trim, sort_by='input_story')

    input_keys = ['input_story', 'input_question']
    output_keys = ['boundary', 'input_answer']
    valid_batch_generator_val = random_generator(data_dict=valid_data, batch_size=batch_size_eval,
                                                 input_keys=input_keys, output_keys=output_keys,
                                                 trim_function=multiword_trim, sort_by='input_story')


    return squad_eval.get_best()
