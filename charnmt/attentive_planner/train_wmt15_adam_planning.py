import os
import glob
from collections import OrderedDict
from nmt import train
from char_planning import *
import re
import json

# layers = {'ff': ('param_init_fflayer', 'fflayer'),
#           'fff': ('param_init_ffflayer', 'ffflayer'),
#           'gru': ('param_init_gru', 'gru_layer'),
#           'planning': ('param_init_planning',
#                               'biscale__planning_decoder_decoder'),
#           }

def main(job_id, params):
    #re_load = False
    #save_file_name = 'bpe2char_biscale_decoder_adam'
    save_file_name = 'planning_10_true'
    source_dataset = params['train_data_path'] + params['source_dataset']
    target_dataset = params['train_data_path'] + params['target_dataset']
    valid_source_dataset = params['dev_data_path'] + params['valid_source_dataset']
    valid_target_dataset = params['dev_data_path'] + params['valid_target_dataset']
    source_dictionary = params['train_data_path'] + params['source_dictionary']
    target_dictionary = params['train_data_path'] + params['target_dictionary']

    decoder_type = params['decoder_type']
    save_file_name = 'bpe2char_{}_adam.last'.format(decoder_type)

    print "We are using th decoder:", decoder_type

    layers = {'ff': ('param_init_fflayer', 'fflayer'),
              'fff': ('param_init_ffflayer', 'ffflayer'),
              'gru': ('param_init_gru', 'gru_layer'),
              '{}'.format(decoder_type): ('param_init_{}'.format(decoder_type),
                                  '{}_decoder'.format(decoder_type)),
              }

    re_load = params['reload'] == 'True'


    print params, params['save_path'], save_file_name

    validerr = train(
        max_epochs=int(params['max_epochs']),
        patience=int(params['patience']),
        dim_word=int(params['dim_word']),
        dim_word_src=int(params['dim_word_src']),
        save_path=params['save_path'],
        save_file_name=save_file_name,
        re_load=re_load,
        enc_dim=int(params['enc_dim']),
        dec_dim=int(params['dec_dim']),
        n_words=int(params['n_words']),
        n_words_src=int(params['n_words_src']),
        decay_c=float(params['decay_c']),
        lrate=float(params['learning_rate']),
        optimizer=params['optimizer'],
        maxlen=int(params['maxlen']),
        maxlen_trg=int(params['maxlen_trg']),
        maxlen_sample=int(params['maxlen_sample']),
        batch_size=int(params['batch_size']),
        valid_batch_size=int(params['valid_batch_size']),
        sort_size=int(params['sort_size']),
        validFreq=int(params['validFreq']),
        dispFreq=int(params['dispFreq']),
        saveFreq=int(params['saveFreq']),
        sampleFreq=int(params['sampleFreq']),
        clip_c=int(params['clip_c']),
        datasets=[source_dataset, target_dataset],
        valid_datasets=[valid_source_dataset, valid_target_dataset],
        dictionaries=[source_dictionary, target_dictionary],
        use_dropout=int(params['use_dropout']),
        source_word_level=int(params['source_word_level']),
        target_word_level=int(params['target_word_level']),
        layers=layers,
        save_every_saveFreq=1,
        save_burn_in=0,
        use_bpe=1,
        init_params=init_params,
        build_model=build_model,
        build_sampler=build_sampler,
        gen_sample=gen_sample,
        c_lb=float(params['c_lb']),
        st_estimator=params['st_estimator'],
        use_gate = params['use_gate'] == 'True',
        learn_t = params['learn_t'] == 'True',
        plan_step=int(params['plan_step']),
        shuffle_dataset = params['shuffle_dataset'] == 'True',
        only_use_w = params['only_use_w'] == 'True',
        nb_cumulate = int(params['nb_cumulate']),
        repeat_actions=params['repeat_actions'] == 'True',
        decoder_type=decoder_type,
        layer_norm=False if not 'layer_norm' in params else
    params['layer_norm'] == 'True',
        planning_do_layerNorm = False if not 'planning_do_layerNorm' in params
        else params['planning_do_layerNorm'] == 'True',
    )
    return validerr

if __name__ == '__main__':

    import sys, time
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'wmt15_deen_bpe2char_adam_planning.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = OrderedDict()

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    main(0, params)
