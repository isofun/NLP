from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import shared
from nmt import (build_sampler, gen_sample)
from theano_util import (numpy_floatX, load_params, init_theano_params)
from util import load_dict, load_config

import argparse, numpy
import logging, ipdb

def prepare_data(seqs_x):
    lengths_x = [len(s) for s in seqs_x]

    new_seqs_x = []
    new_lengths_x = []
    for (l_x, s_x) in zip(lengths_x, seqs_x):
        new_seqs_x.append(s_x)
        new_lengths_x.append(l_x)
    lengths_x = new_lengths_x
    seqs_x = new_seqs_x

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1

    x = numpy.zeros((1, maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[:, :lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
    return x, x_mask

def main(model, dictionary, dictionary_target, source_file, saveto, k=5, batch_size = 1, opt_base=None,
         normalize=False, output_attention=False):
    trng = RandomStreams(1234)
    use_noise = shared(numpy.float32(0.))

    #load params
    if opt_base is None:
        options = load_config(model)
    else:
        options = load_config(opt_base)

    param_list = numpy.load(model).files
    param_list = dict.fromkeys(
        [key for key in param_list if not key.startswith('adam_')], 0)
    params = load_params(model, param_list, '')
    tparams = init_theano_params(params)

    #load dictionary
    if dictionary is None:
        dictionary = options['dictionaries'][0]
    word_dict = load_dict(dictionary)

    if options['n_words_src']:
        for key, idx in word_dict.items():
            if idx >= options['n_words_src']:
                del word_dict[key]
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    if dictionary_target is None:
        dictionary_target = options['dictionaries'][1]
    word_dict_trg = load_dict(dictionary_target)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    def _send_jobs(fname):
        retval = []
        retval_ori = []
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                words = line.strip().split()
                if len(words) == 0:
                    continue
                retval_ori.append(line.strip())
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                retval.append(x)
        logging.info('total %s sentences' % len(retval))
        return retval, retval_ori

    sources, sources_ori = _send_jobs(source_file)

    batches = []
    for i in range(len(sources) / batch_size):
        batches.append(prepare_data(sources[i * batch_size: (i + 1) * batch_size]))
    if (i + 1) * batch_size < len(sources):
        batches.append(prepare_data(sources[(i + 1) * batch_size: ]))
    final_sentences = []
    f_init, f_next = build_sampler(tparams, options, use_noise, trng)

    samples, scores, word_probs, alignment, _ = gen_sample([f_init], [f_next],
                                                   batch[69][0],
                                                   trng=trng, k=k, maxlen=200,
                                                   stochastic=False, argmax=False,
                                                   return_alignment=True)
    if normalize:
        lengths = numpy.array([len(s) for s in samples])
        scores = scores / lengths
    final_words = samples[numpy.argmin(scores)]
    final_sentences.append(' '.join([word_idict_trg[w] for w in final_words]) + '\n')

    ipdb.set_trace()
    # with open(saveto, 'w') as fout:
    #     for sentence in final_sentences:
    #         fout.write(sentence)
    # print 'Done'

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    parser = argparse.ArgumentParser()

    # k: beam size
    parser.add_argument('-k', type=int, default=5, help='beam size')
    # n: if normalize
    parser.add_argument('-n', action="store_true", default=True, help = 'normalize')
    # # if output attention
    # parser.add_argument('-a', action="store_true", default=False)
    # option
    parser.add_argument('-o', type=str, default=None, help = 'option base')

    # source side dictionary
    parser.add_argument('--dictionary', type=str, default=None)
    # target side dictionary
    parser.add_argument('--dictionary_target', type=str, default=None)
    # model.npz
    parser.add_argument('model', type=str, help = 'model')
    # source file
    parser.add_argument('source', type=str)
    # translation file
    parser.add_argument('saveto', type=str)


    args = parser.parse_args()

    import datetime
    start_time = datetime.datetime.now()

    main(args.model, args.dictionary, args.dictionary_target, args.source,
         args.saveto, k=args.k, opt_base=args.o, normalize=args.n)

    print 'Elapsed Time: %s' % str(datetime.datetime.now() - start_time)