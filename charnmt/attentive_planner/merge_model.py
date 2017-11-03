
import numpy
import cPickle as pkl
import argparse
from char_planning import (build_sampler, gen_sample, init_params)
from mixer import *
import os
import glob


def add_load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] += pp[kk]

    return params


def get_all_last(folder, nb, from_no = -1):
    files = glob.glob(folder+"/*best.npz")
    #files = glob.glob(folder+"/*.npz")
    files.sort(key=os.path.getmtime)

    file_tmp = []

    print ""
    if from_no >= 0:
        for f in files:

            if 'grads' not in f:
                file_tmp.append(f)

            print '{}'.format(from_no), f
            if '{}'.format(from_no) in f:
                print "trouve!"
                break

        files = file_tmp

    files = files[-nb:]

    print files

    return list(files)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True)
    parser.add_argument('-s', type=str)
    parser.add_argument('-p', type=str)
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('-f', type=int, default=-1)
    # THEANO_FLAGS='device=cpu' python merge_model.py -f 1035000 -n 5 -d ~/tmplisa4/attentive_model/NIPS2_csen_norepeat_ln

    args = parser.parse_args()

    folder = args.d
    short = args.s
    nb = args.n
    savepath = args.p
    from_no = args.f

    #getting the last models...
    models = get_all_last(folder, nb, from_no)
    nb = len(models)
    if not short:
        short = models[-1].split('/')[-1].replace('.', '_') + "_merged.last.npz"

    if not savepath:
        savepath = folder

    saveto = savepath + short

    print "merging the last {} best of {} and putting them in {}".format(nb, folder, short)


    all_params = None
    options = None

    print "Collecting the models..."
    for no, model in enumerate(models):

        # load model model_options
        pkl_file = model.split('.')[0] + '.last.pkl'
        with open(pkl_file, 'rb') as f:
            options = pkl.load(f)


        # collect al the params
        if no == 0:
            # allocate model parameters
            all_params = init_params(options)
            all_params = load_params(model, all_params)
        else:
            all_params = add_load_params(model, all_params)


    # get the mean
    print "Averaging..."
    for kk, vv in all_params.iteritems():
        all_params[kk] = all_params[kk] / nb

    print "saving to...", saveto
    np.savez(saveto, **all_params)
    pkl.dump(options, open(saveto.split('.')[0] + '.last.pkl', 'wb'))
