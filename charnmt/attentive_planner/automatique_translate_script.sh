#!/bin/bash
# -*- coding: utf-8 -*-

# Will translate a given model on all datasets, for a given laguage.
# The path to the dataset needs to be changed.

# An example of command:
# bash automatique_translate_script.sh csen_model_1 model.npz csen 15 False
# Meaning: use model model.npz, for English to Czech, with beam size 15, using planning mechanism.
# In the resulting file, name it csen_model_1

SHORT=$1
FILE=$2
MODE=$3
BEAM=$4
BASE=$5

OUT="None"
SOURCES="NONE"
TARGETS="NONE"
SOURCEDIC="NONE"
TARGETDIC="NONE"
NAMES="NONE"


if [ "$MODE" == "deen" ]; then

    echo "Going to evaluate on the deen datasets"
    OUT="all_translation/deen/"
    SOURCES=("/data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/dev/newstest2013.en.tok.bpe" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/dev/newstest2014-deen-ref.en.tok.bpe" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/testset_2015/test_2015/newstest2015-ende-src.en.tok.bpe")
    TARGETS=("/data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/dev/newstest2013.de.tok" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/dev/newstest2014-deen-ref.de.tok" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/testset_2015/test_2015/newstest2015-ende-ref.de.tok")
    NAMES=("dev" "test1" "test2")

    SOURCEDIC="/data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/train/all_de-en.en.tok.bpe.word.pkl"
    TARGETDIC="/data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/train/all_de-en.de.tok.300.pkl"

elif [ "$MODE" == "fien" ]; then

    echo "Going to evaluate on the fien datasets"
    OUT="all_translation/fien/"
    SOURCES=("/data/lisatmp4/gulcehrc/nmt/data/wmt15/fien/dev/newsdev2015-enfi-src.en.tok.bpe" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/testset_2015/test_2015/newstest2015-enfi-src.en.tok.bpe")
    TARGETS=("/data/lisatmp4/gulcehrc/nmt/data/wmt15/fien/dev/newsdev2015-enfi-ref.fi.tok" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/testset_2015/test_2015/newstest2015-enfi-ref.fi.tok")
    NAMES=("dev" "test1")

    SOURCEDIC="/data/lisatmp4/gulcehrc/nmt/data/wmt15/fien/train/all_fi-en.en.tok.bpe.word.pkl"
    TARGETDIC="/data/lisatmp4/gulcehrc/nmt/data/wmt15/fien/train/all_fi-en.fi.tok.300.pkl"

elif [ "$MODE" == "csen" ]; then

    echo "Going to evaluate on the csen datasets"
    OUT="all_translation/csen/"
    SOURCES=("/data/lisatmp4/gulcehrc/nmt/data/wmt15/csen/dev/newstest2013-src.en.tok.bpe" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/csen/dev/newstest2014-csen-src.en.tok.bpe" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/testset_2015/test_2015/newstest2015-encs-src.en.tok.bpe")
    TARGETS=("/data/lisatmp4/gulcehrc/nmt/data/wmt15/csen/dev/newstest2013-ref.cs.tok" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/csen/dev/newstest2014-csen-ref.cs.tok" "/data/lisatmp4/gulcehrc/nmt/data/wmt15/testset_2015/test_2015/newstest2015-encs-ref.cs.tok")
    NAMES=("dev" "test1" "test2")

    SOURCEDIC="/data/lisatmp4/gulcehrc/nmt/data/wmt15/csen/train/all_cs-en.en.tok.bpe.word.pkl"
    TARGETDIC="/data/lisatmp4/gulcehrc/nmt/data/wmt15/csen/train/all_cs-en.cs.tok.300.pkl"

else

    echo "Mode $MODE unknown"
    exit 1

fi

echo "Small summary:"
echo "We will evaluate the model $SHORT ($FILE, language $MODE), with beam size $BEAM, and put the results in $OUT"
for i in $(seq 1 ${#SOURCES[@]}); do


    SOURCE=${SOURCES[$i - 1]}
    TARGET=${TARGETS[$i - 1]}
    NAME=${NAMES[$i - 1]}

    echo "beepbop, doing $SOURCE"

    OFILE=$OUT'/'$SHORT'_translation_b'$BEAM'_'$NAME'.txt'

    if [ ! -f "$OFILE" ]; then
            echo "choochoo $OFILE"

                    #THEANO_FLAGS='floatX=float32, device=gpu, optimizer=fast_run, optimizer_including=alloc_empty_to_zeros, lib.cnmem=1' python translate_planning_gpu.py -utf8 -k 15 -n -dec_c /u/dutilfra/data/d073e9ab-8eb3-46c6-a304-12172e4585a0/deen_norepeat_ln/bpe2char_gru_planning_adam.last.285000.best.npz /data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/train/all_de-en.en.tok.bpe.word.pkl /data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/train/all_de-en.de.tok.300.pkl /data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/dev/newstest2014-deen-ref.en.tok.bpe models/NIPS_2014_deen_norepeat_ln_285000_b15.txt
            COMMAND = "NONE"

            if [ $BASE == "True" ]; then
                COMMAND="THEANO_FLAGS='floatX=float32, device=gpu, optimizer=fast_run, optimizer_including=alloc_empty_to_zeros, lib.cnmem=1' python translate_planning_gpu.py -utf8 -k $BEAM -n -dec_c -base $FILE $SOURCEDIC $TARGETDIC $SOURCE $OFILE"
            else
                COMMAND="THEANO_FLAGS='floatX=float32, device=gpu, optimizer=fast_run, optimizer_including=alloc_empty_to_zeros, lib.cnmem=1' python translate_planning_gpu.py -utf8 -k $BEAM -n -dec_c $FILE $SOURCEDIC $TARGETDIC $SOURCE $OFILE"
            fi

            echo $COMMAND
            #COMMAND="THEANO_FLAGS='device=cpu' python translate_planning.py -utf8 -p 10 -k $BEAM -n -dec_c $STR  /data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/train/all_de-en.en.tok.bpe.word.pkl /data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/train/all_de-en.de.tok.300.pkl /data/lisatmp4/gulcehrc/nmt/data/wmt15/deen/dev/newstest2013.en.tok.bpe $OFILE"
	   	    eval $COMMAND

	   	    #The bleu score
	   	    BLEU="perl ../preprocess/multi-bleu.perl $TARGET  < $OFILE"
	   	    echo $OFILE >> $OUT"/bleu_score.txt"
	   	    eval $BLEU >> $OUT"/bleu_score.txt"


    else
        echo "$OFILE was already made!"
    fi

done

echo "Done!"
