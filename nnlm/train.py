import numpy
import os
import sys
from nmt import train


def main(job_id, params):
    # print params



    validerr = train(saveto=params['saveto'][0],
                     loadfrom = params['loadfrom'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words_src=params['n-words-src'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     maxlen=50,
                     batch_size=80,
                     valid_batch_size=80,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=20000,
                     sampleFreq=100,
                     max_epochs=5000,  # max iteration
                     patience=1000,  # early stop patience with BLEU score
                     finish_after=1000000,  # max updates
                     datasets=['../data/cn.1w_with_unk.txt'],
                     valid_datasets=['../NIST/MT02/en0'],
                     dictionaries=['../data/en.txt.shuf.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False,
                     **bleuvalid_params)

    return validerr

if __name__ == '__main__':

    import datetime
    start_time = datetime.datetime.now()
    main(0, {
        'loadfrom': ['model_core_10kto10k_69w.npz'],
        'saveto': [],
        'dim_word': [512],
        'dim': [1024],
        'n-words-src': [10000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True]})

    print 'Elapsed Time: %s' % str(datetime.datetime.now() - start_time)

