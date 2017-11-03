Command for using translate.py BPE-case:
python translate.py -k {beam_width} -p {number_of_processors} -n -bpe {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}

Command for using translate.py Char-case:
python translate.py -k {beam_width} -p {number_of_processors} -n -dec_c -utf8 {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}

Command for using translate_both.py BPE-case:
python translate_both.py -k {beam_width} -p {number_of_processors} -n -bpe {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}

Command for using translate_both.py Char-case:
python translate_both.py -k {beam_width} -p {number_of_processors} -n -dec_c -utf8 {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}

Command for using translate_attc.py Char-case:
python translate_attc.py -k {beam_width} -p {number_of_processors} -n -dec_c -utf8 {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}

Command for using `multi-bleu.perl':

#multi-bleu.perl /Tmp/dutilfra/nmt/data/wmt15/deen/dev/newstest2013.de.tok < attentive_planner/models/translate_baseline_both_18.txt
#perl multi-bleu.perl /data/lisatmp4/gulcehrc/nmt/data/wmt15/fien/dev/newsdev2015-enfi-ref.fi.tok < attentive_planner/models/translate_fien_180000.txt
#perl multi-bleu.perl /data/lisatmp4/gulcehrc/nmt/data/wmt15/csen/dev/newstest2013-ref.cs.tok < attentive_planner/models/NIPS_csen_repeat_1035000_b15.txt

perl multi-bleu.perl {reference.txt} < {translated.txt}
