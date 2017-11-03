import numpy
import os

from da2qgen import train

def main(job_id, params):
    data_path = "/data/lisatmp4/gulcehrc/data/squad/"
    datasets = [os.path.join(data_path, 'train-v1.1.json'),
                os.path.join(data_path, 'dev-v1.1.json')]

    debug = False
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words_q=params['n_words_q'],
                     n_words_doc=params['n_words_doc'],
                     n_words_ans=params['n_words_ans'],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     use_pointer_softmax=True,
                     debug=debug,
                     use_batch_norm=True,
                     patience=1000,
                     datasets=datasets,
                     batch_size=16,#32
                     temp_switch=params['temp_switch'],
                     valid_batch_size=16,
                     validFreq=1200,
                     use_doc_emb_ans=params['use_doc_emb_ans'],
                     dispFreq=20,
                     maxlen=660,
                     assign_priority_point=True,
                     saveFreq=30000,
                     sampleFreq=600,
                     use_dropout=params['use-dropout'][0],
                     do_planning=False,
                     val_size=2000,
                     )
    return validerr


if __name__ == '__main__':
    m_path = "/data/lisatmp4/dutilfra/d2qgen/baseline5/"
    main(0, {
        'model': [m_path + \
                  'model_pointer_softmax_model_temp_switch=1_uadam_qgenS.npz'],
        'dim_word': [600],
        'dim': [800],
        'n_words_doc': 64000,
        'n_words_ans': 30000,
        'n_words_q': 2000,
        'optimizer': ['adadelta'],
        'decay-c': [5e-5],
        'clip-c': [1.2],
        'use-dropout': [True],
        'learning-rate': [0.0002],
        'temp_switch': 1.0,
        'use_doc_emb_ans': True,
        'reload': [True]})
