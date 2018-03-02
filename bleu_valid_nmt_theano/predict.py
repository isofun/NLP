#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import json
import numpy
import copy
import argparse

import os
import sys
import time
import logging
import ipdb

import itertools

from subprocess import Popen

from collections import OrderedDict

from bleu_validator import BleuValidator

profile = False

from data_iterator import TextIterator
from training_progress import TrainingProgress
from util import *
from theano_util import *
from alignment_util import *
from raml_distributions import *

from layers import *
from initializers import *
from optimizers import *
from metrics.scorer_provider import ScorerProvider

from domain_interpolation_data_iterator import DomainInterpolatorTextIterator

# batch preparation
def prepare_data(seqs_x, seqs_y, weights=None, maxlen=None, n_words_src=30000,
                 n_words=30000, n_factors=1):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        new_weights = []
        if weights is None:
            weights = [None] * len(seqs_y) # to make the zip easier
        for l_x, s_x, l_y, s_y, w in zip(lengths_x, seqs_x, lengths_y, seqs_y, weights):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_weights.append(w)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        weights = new_weights

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            if weights is not None:
                return None, None, None, None, None
            else:
                return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype(floatX)
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype(floatX)
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.
    if weights is not None:
        return x, x_mask, y, y_mask, weights
    else:
        return x, x_mask, y, y_mask

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params = get_layer_param('embedding')(options, params, options['n_words_src'], options['dim_per_factor'], options['factors'], suffix='')
    if not options['tie_encoder_decoder_embeddings']:
        params = get_layer_param('embedding')(options, params, options['n_words'], options['dim_word'], suffix='_dec')

    # encoder: bidirectional RNN
    params = get_layer_param(options['encoder'])(options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              recurrence_transition_depth=options['enc_recurrence_transition_depth'])
    params = get_layer_param(options['encoder'])(options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              recurrence_transition_depth=options['enc_recurrence_transition_depth'])
    if options['enc_depth'] > 1:
        for level in range(2, options['enc_depth'] + 1):
            prefix_f = pp('encoder', level)
            prefix_r = pp('encoder_r', level)

            if level <= options['enc_depth_bidirectional']:
                params = get_layer_param(options['encoder'])(options, params,
                                                             prefix=prefix_f,
                                                             nin=options['dim'],
                                                             dim=options['dim'],
                                                             recurrence_transition_depth=options['enc_recurrence_transition_depth'])
                params = get_layer_param(options['encoder'])(options, params,
                                                             prefix=prefix_r,
                                                             nin=options['dim'],
                                                             dim=options['dim'],
                                                             recurrence_transition_depth=options['enc_recurrence_transition_depth'])
            else:
                params = get_layer_param(options['encoder'])(options, params,
                                                             prefix=prefix_f,
                                                             nin=options['dim'] * 2,
                                                             dim=options['dim'] * 2,
                                                             recurrence_transition_depth=options['enc_recurrence_transition_depth'])


    ctxdim = 2 * options['dim']

    dec_state = options['dim']
    if options['decoder'].startswith('lstm'):
        dec_state *= 2



    # readout
    params = get_layer_param('ff')(options, params, prefix='ff_logit_lstm',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)

    params = get_layer_param('ff')(options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'],
                                weight_matrix = not options['tie_decoder_embeddings'],
                                followed_by_softmax=True)

    return params


# bidirectional RNN encoder: take input x (optionally with mask), and produce sequence of context vectors (ctx)
def build_encoder(tparams, options, dropout, x_mask=None, sampling=False):

    x = tensor.tensor3('x', dtype='int64')
    # source text; factors 1; length 5; batch size 10
    x.tag.test_value = (numpy.random.rand(1, 5, 10)*100).astype('int64')

    # for the backward rnn, we just need to invert x
    xr = x[:,::-1]
    if x_mask is None:
        xr_mask = None
    else:
        xr_mask = x_mask[::-1]

    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    # word embedding for forward rnn (source)
    emb = get_layer_constr('embedding')(tparams, x, suffix='', factors= options['factors'])

    # word embedding for backward rnn (source)
    embr = get_layer_constr('embedding')(tparams, xr, suffix='', factors= options['factors'])

    if options['use_dropout']:
        source_dropout = dropout((n_timesteps, n_samples, 1), options['dropout_source'])
        if not sampling:
            source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
        emb *= source_dropout

        if sampling:
            embr *= source_dropout
        else:
            # we drop out the same words in both directions
            embr *= source_dropout[::-1]


    ## level 1
    proj = get_layer_constr(options['encoder'])(tparams, emb, options, dropout,
                                                prefix='encoder',
                                                mask=x_mask,
                                                dropout_probability_below=options['dropout_embedding'],
                                                dropout_probability_rec=options['dropout_hidden'],
                                                recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                truncate_gradient=options['encoder_truncate_gradient'],
                                                profile=profile)
    projr = get_layer_constr(options['encoder'])(tparams, embr, options, dropout,
                                                 prefix='encoder_r',
                                                 mask=xr_mask,
                                                 dropout_probability_below=options['dropout_embedding'],
                                                 dropout_probability_rec=options['dropout_hidden'],
                                                 recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                 truncate_gradient=options['encoder_truncate_gradient'],
                                                 profile=profile)

    # discard LSTM cell state
    if options['encoder'].startswith('lstm'):
        proj[0] = get_slice(proj[0], 0, options['dim'])
        projr[0] = get_slice(projr[0], 0, options['dim'])

    ## bidirectional levels before merge
    for level in range(2, options['enc_depth_bidirectional'] + 1):
        prefix_f = pp('encoder', level)
        prefix_r = pp('encoder_r', level)

        # run forward on previous backward and backward on previous forward
        input_f = projr[0][::-1]
        input_r = proj[0][::-1]

        proj = get_layer_constr(options['encoder'])(tparams, input_f, options, dropout,
                                                    prefix=prefix_f,
                                                    mask=x_mask,
                                                    dropout_probability_below=options['dropout_hidden'],
                                                    dropout_probability_rec=options['dropout_hidden'],
                                                    recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                    truncate_gradient=options['encoder_truncate_gradient'],
                                                    profile=profile)
        projr = get_layer_constr(options['encoder'])(tparams, input_r, options, dropout,
                                                     prefix=prefix_r,
                                                     mask=xr_mask,
                                                     dropout_probability_below=options['dropout_hidden'],
                                                     dropout_probability_rec=options['dropout_hidden'],
                                                     recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                     truncate_gradient=options['encoder_truncate_gradient'],
                                                     profile=profile)

        # discard LSTM cell state
        if options['encoder'].startswith('lstm'):
            proj[0] = get_slice(proj[0], 0, options['dim'])
            projr[0] = get_slice(projr[0], 0, options['dim'])

        # residual connections
        if level > 1:
            proj[0] += input_f
            projr[0] += input_r

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    ## forward encoder layers after bidirectional layers are concatenated
    for level in range(options['enc_depth_bidirectional'] + 1, options['enc_depth'] + 1):

        ctx += get_layer_constr(options['encoder'])(tparams, ctx, options, dropout,
                                                   prefix=pp('encoder', level),
                                                   mask=x_mask,
                                                   dropout_probability_below=options['dropout_hidden'],
                                                   dropout_probability_rec=options['dropout_hidden'],
                                                   recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                   truncate_gradient=options['encoder_truncate_gradient'],
                                                   profile=profile)[0]

    return x, ctx


# RNN decoder (including embedding and feedforward layer before output)
def build_decoder(tparams, options, y, ctx, dropout, x_mask=None, y_mask=None, sampling=False, pctx_=None, shared_vars=None):

    # tell RNN whether to advance just one step at a time (for sampling),
    # or loop through sequence (for training)
    if sampling:
        one_step=True
    else:
        one_step=False

    if options['use_dropout']:
        if sampling:
            target_dropout = dropout(dropout_probability=options['dropout_target'])
        else:
            n_timesteps_trg = y.shape[0]
            n_samples = y.shape[1]
            target_dropout = dropout((n_timesteps_trg, n_samples, 1), options['dropout_target'])
            target_dropout = tensor.tile(target_dropout, (1, 1, options['dim_word']))

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings'] else '_dec'
    emb = get_layer_constr('embedding')(tparams, y, suffix=decoder_embedding_suffix)
    if options['use_dropout']:
        emb *= target_dropout

    if sampling:
        emb = tensor.switch(y[:, None] < 0,
            tensor.zeros((1, options['dim_word'])),
            emb)
    else:
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted


    # hidden layer taking RNN state, previous word embedding and context vector as input
    # (this counts as the first layer in our deep output, which is always on)
    logit_lstm = get_layer_constr('ff')(tparams, ctx, options, dropout,
                                    dropout_probability=options['dropout_hidden'],
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options, dropout,
                                    dropout_probability=options['dropout_embedding'],
                                    prefix='ff_logit_prev', activ='linear')

    logit = tensor.tanh(logit_lstm+logit_prev)

    # last layer
    logit_W = tparams['Wemb' + decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None
    logit = get_layer_constr('ff')(tparams, logit, options, dropout,
                            dropout_probability=options['dropout_hidden'],
                            prefix='ff_logit', activ='linear', W=logit_W, followed_by_softmax=True)

    return logit

# build a training model
def build_model(tparams, options):

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy_floatX(0.))
    dropout = dropout_constr(options, use_noise, trng, sampling=False)

    x_mask = tensor.matrix('x_mask', dtype=floatX)
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype=floatX)
    # source text length 5; batch size 10
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype(floatX)
    # target text length 8; batch size 10
    y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
    y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype(floatX)

    x, ctx = build_encoder(tparams, options, dropout, x_mask, sampling=False)

    logit = build_decoder(tparams, options, y, ctx, dropout, x_mask=x_mask, y_mask=y_mask, sampling=False)

    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    #print "Print out in build_model()"
    #print opt_ret
    return trng, use_noise, x, x_mask, y, y_mask, cost


# build a sampler
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):

    dropout = dropout_constr(options, use_noise, trng, sampling=True)

    x, ctx = build_encoder(tparams, options, dropout, x_mask=None, sampling=True)



    logging.info('Building f_init...')
    outs = [ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    logging.info('Done')

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    y.tag.test_value = -1 * numpy.ones((10,)).astype('int64')

    ctx_now = tensor.matrix('ctx_now', dtype=floatX)

    logit = build_decoder(tparams, options, y, ctx_now, dropout, x_mask=None, y_mask=None, sampling=True)

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    logging.info('Building f_next..')
    inps = [y, ctx_now]
    outs = [next_probs, next_sample]

    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    logging.info('Done')

    return f_init, f_next

# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1 and argmax:
        assert not stochastic, \
            'Beam search does not support stochastic sampling with argmax'

    sample = []
    sample_score = []
    sample_word_probs = []
    if stochastic:
        if argmax:
            sample_score = 0
        live_k=k
    else:
        live_k = 1

    dead_k = 0

    hyp_samples=[ [] for i in xrange(live_k) ]
    word_probs=[ [] for i in xrange(live_k) ]
    hyp_scores = numpy.zeros(live_k).astype(floatX)

    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    ctx0 = [None]*num_models
    next_p = [None]*num_models
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        ret = f_init[i](x)

        # to more easily manipulate batch size, go from (layers, batch_size, dim) to (batch_size, layers, dim)
        ctx0[i] = ret[0]

    next_w = -1 * numpy.ones((live_k,)).astype('int64')  # bos indicator

    # x is a sequence of word ids followed by 0, eos id
    for ii in xrange(numpy.minimum(x.shape[1], maxlen)):
        for i in xrange(num_models):
            ctx = numpy.tile(ctx0[i], [live_k, 1])

            # for theano function, go from (batch_size, layers, dim) to (layers, batch_size, dim)

            inps = [next_w, ctx[ii]]
            ret = f_next[i](*inps)

            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_p[i], next_w_tmp = ret[0], ret[1]

            # to more easily manipulate batch size, go from (layers, batch_size, dim) to (batch_size, layers, dim)

        if stochastic:
            #batches are not supported with argmax: output data structure is different
            if argmax:
                nw = sum(next_p)[0].argmax()
                sample.append(nw)
                sample_score += numpy.log(next_p[0][0, nw])
                if nw == 0:
                    break
            else:
                #FIXME: sampling is currently performed according to the last model only
                nws = next_w_tmp
                cand_scores = numpy.array(hyp_scores)[:, None] - numpy.log(next_p[-1])
                probs = next_p[-1]

                for idx,nw in enumerate(nws):
                    hyp_samples[idx].append(nw)


                for ti in xrange(live_k):
                    hyp_scores[ti]=cand_scores[ti][nws[ti]]
                    word_probs[ti].append(probs[ti][nws[ti]])

                new_hyp_samples=[]
                new_hyp_scores=[]
                new_word_probs=[]
                for hyp_sample, hyp_score, hyp_word_prob in zip(hyp_samples,hyp_scores, word_probs):
                    if hyp_sample[-1]  > 0:
                        new_hyp_samples.append(copy.copy(hyp_sample))
                        new_hyp_scores.append(hyp_score)
                        new_word_probs.append(hyp_word_prob)
                    else:
                        sample.append(copy.copy(hyp_sample))
                        sample_score.append(hyp_score)
                        sample_word_probs.append(hyp_word_prob)

                hyp_samples=new_hyp_samples
                hyp_scores=new_hyp_scores
                word_probs=new_word_probs

                live_k=len(hyp_samples)
                if live_k < 1:
                    break

                next_w = numpy.array([w[-1] for w in hyp_samples])
        else:
            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
            probs = sum(next_p)/num_models
            cand_flat = cand_scores.flatten()
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]

            voc_size = next_p[0].shape[1]
            # index of each k-best hypothesis
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype(floatX)
            new_word_probs = []

            # ti -> index of k-best hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])


            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            word_probs = []

            # sample and sample_score hold the k-best translations and their scores
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(copy.copy(new_hyp_samples[idx]))
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(copy.copy(new_hyp_samples[idx]))
                    hyp_scores.append(new_hyp_scores[idx])
                    word_probs.append(new_word_probs[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])

    # dump every remaining one
    if not argmax and live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])
            sample_word_probs.append(word_probs[idx])


    return sample, sample_score, sample_word_probs


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalization_alpha=0.0):
    probs = []
    n_done = 0

    alignments_json = []

    for x, y in iterator:
        #ensure consistency in number of factors
        if len(x[0][0]) != options['factors']:
            logging.error('Mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(options['factors'], len(x[0][0])))
            sys.exit(1)

        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'],
                                            n_factors=options['factors'])

        ### in optional save weights mode.
        pprobs = f_log_probs(x, x_mask, y, y_mask)

        # normalize scores according to output length
        if normalization_alpha:
            adjusted_lengths = numpy.array([numpy.count_nonzero(s) ** normalization_alpha for s in y_mask.T])
            pprobs /= adjusted_lengths

        for pp in pprobs:
            probs.append(pp)

        logging.debug('%d samples computed' % (n_done))

    return numpy.array(probs)



def train(dim_word=512,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          enc_depth=1, # number of layers in the encoder
          dec_depth=1, # number of layers in the decoder
          enc_recurrence_transition_depth=1, # number of GRU transition operations applied in the encoder. Minimum is 1. (Only applies to gru)
          dec_base_recurrence_transition_depth=2, # number of GRU transition operations applied in the first layer of the decoder. Minimum is 2. (Only applies to gru_cond)
          dec_high_recurrence_transition_depth=1, # number of GRU transition operations applied in the higher layers of the decoder. Minimum is 1. (Only applies to gru)
          dec_deep_context=False, # include context vectors in deeper layers of the decoder
          enc_depth_bidirectional=None, # first n encoder layers are bidirectional (default: all)
          factors=1, # input factors
          dim_per_factor=None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
          encoder='gru',
          decoder='gru_cond',
          decoder_deep='gru',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=1000,
          decay_c=0.,  # L2 regularization penalty
          map_decay_c=0., # L2 regularization penalty towards original weights
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.0001,  # learning rate
          n_words_src=None,  # source vocabulary size
          n_words=None,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='adam',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          loadfrom = '',
          validFreq=10000,
          saveFreq=30000,   # save the parameters after every saveFreq updates
          decodeFreq = 10000,
          sampleFreq=10000,   # generate some samples after every sampleFreq
          datasets=[ # path to training datasets (source and target)
              None,
              None],
          valid_datasets=[None, # path to validation datasets (source and target)
                          None],
          dictionaries=[ # path to dictionaries (json file created with ../data/build_dictionary.py). One dictionary per input factor; last dictionary is target-side dictionary.
              None,
              None],
          use_dropout=False,
          dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
          dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
          dropout_source=0, # dropout source words (0: no dropout)
          dropout_target=0, # dropout target words (0: no dropout)
          reload_=False,
          ignore_list = '',
          reload_training_progress=True, # reload trainig progress (only used if reload_ is True)
          overwrite=False,
          # external_validation_script=None,
          shuffle_each_epoch=True,
          sort_by_length=True,
          use_domain_interpolation=False, # interpolate between an out-domain training corpus and an in-domain training corpus
          domain_interpolation_min=0.1, # minimum (initial) fraction of in-domain training data
          domain_interpolation_max=1.0, # maximum fraction of in-domain training data
          domain_interpolation_inc=0.1, # interpolation increment to be applied each time patience runs out, until maximum amount of interpolation is reached
          domain_interpolation_indomain_datasets=[None, None], # in-domain parallel training corpus (source and target)
          anneal_restarts=3, # when patience run out, restart with annealed learning rate X times before early stopping
          anneal_decay=0.5, # decay learning rate by this amount on each restart
          maxibatch_size=20, #How many minibatches to load at one time
          objective="CE", #CE: cross-entropy; MRT: minimum risk training (see https://www.aclweb.org/anthology/P/P16/P16-1159.pdf) \
                          #RAML: reward-augmented maximum likelihood (see https://papers.nips.cc/paper/6547-reward-augmented-maximum-likelihood-for-neural-structured-prediction.pdf)
          mrt_alpha=0.005,
          mrt_samples=100,
          mrt_samples_meanloss=10,
          mrt_reference=False,
          mrt_loss="SENTENCEBLEU n=4", # loss function for minimum risk training
          mrt_ml_mix=0, # interpolate mrt loss with ML loss
          raml_tau=0.85, # in (0,1] 0: becomes equivalent to ML
          raml_samples=1,
          raml_reward="hamming_distance",
          model_version=0.1, #store version used for training for compatibility
          prior_model=None, # Prior model file, used for MAP
          tie_encoder_decoder_embeddings=False, # Tie the input embeddings of the encoder and the decoder (first factor only)
          tie_decoder_embeddings=False, # Tie the input embeddings of the decoder with the softmax output embeddings
          encoder_truncate_gradient=-1, # Truncate BPTT gradients in the encoder to this value. Use -1 for no truncation
          decoder_truncate_gradient=-1, # Truncate BPTT gradients in the decoder to this value. Use -1 for no truncation
          layer_normalisation=False, # layer normalisation https://arxiv.org/abs/1607.06450
          weight_normalisation=False, # normalize weights
          **bleuvalid_params
    ):
    # Model options
    model_options = OrderedDict(sorted(locals().copy().items()))
    bleu_validator = BleuValidator(model_options, **bleuvalid_params)

    if model_options['dim_per_factor'] == None:
        if factors == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            logging.error('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert(len(dictionaries) == factors + 1) # one dictionary per source factor + 1 for target factor
    assert(len(model_options['dim_per_factor']) == factors) # each factor embedding has its own dimensionality
    assert(sum(model_options['dim_per_factor']) == model_options['dim_word']) # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector
    assert(prior_model != None and (os.path.exists(prior_model)) or (map_decay_c==0.0)) # MAP training requires a prior model file: Use command-line option --prior_model

    assert(enc_recurrence_transition_depth >= 1) # enc recurrence transition depth must be at least 1.
    assert(dec_base_recurrence_transition_depth >= 2) # dec base recurrence transition depth must be at least 2.
    assert(dec_high_recurrence_transition_depth >= 1) # dec higher recurrence transition depth must be at least 1.

    if model_options['enc_depth_bidirectional'] is None:
        model_options['enc_depth_bidirectional'] = model_options['enc_depth']
    # first layer is always bidirectional; make sure people don't forget to increase enc_depth as well
    assert(model_options['enc_depth_bidirectional'] >= 1 and model_options['enc_depth_bidirectional'] <= model_options['enc_depth'])

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    if n_words_src is None:
        n_words_src = max(worddicts[0].values()) + 1
        model_options['n_words_src'] = n_words_src
    if n_words is None:
        n_words = max(worddicts[-1].values()) + 1
        model_options['n_words'] = n_words


    logging.info(model_options)


    # initialize training progress
    training_progress = TrainingProgress()
    best_p = None
    best_opt_p = None

    training_progress.bad_counter = 0
    training_progress.anneal_restarts_done = 0
    training_progress.uidx = 0
    training_progress.eidx = 0
    training_progress.estop = False
    training_progress.history_errs = []
    # reload training progress
    training_progress_file = loadfrom + '.progress.json'
    if reload_ and reload_training_progress and os.path.exists(training_progress_file):
        logging.info('Reloading training progress')
        training_progress.load_from_json(training_progress_file)
        if (training_progress.estop == True) or (training_progress.eidx > max_epochs) or (training_progress.uidx >= finish_after):
            logging.warning('Training is already complete. Disable reloading of training progress (--no_reload_training_progress) or remove or modify progress file (%s) to train anyway.' % training_progress_file)
            return numpy.inf

    # adjust learning rate if we resume process that has already entered annealing phase
    if training_progress.anneal_restarts_done > 0:
        lrate *= anneal_decay**training_progress.anneal_restarts_done
        decodeFreq *= anneal_decay ** training_progress.anneal_restarts_done

    logging.info('Loading data')
    train = TextIterator(datasets[0], datasets[1],
                     dictionaries[:-1], dictionaries[-1],
                     n_words_source=n_words_src, n_words_target=n_words,
                     batch_size=batch_size,
                     maxlen=maxlen,
                     skip_empty=True,
                     shuffle_each_epoch=shuffle_each_epoch,
                     sort_by_length=sort_by_length,
                     use_factor=(factors > 1),
                     maxibatch_size=maxibatch_size)

    if valid_datasets and validFreq:
        valid = TextIterator(valid_datasets[0], valid_datasets[1],
                            dictionaries[:-1], dictionaries[-1],
                            n_words_source=n_words_src, n_words_target=n_words,
                            batch_size=valid_batch_size,
                            use_factor=(factors>1),
                            maxlen=maxlen)
    else:
        valid = None

    comp_start = time.time()

    logging.info('Building model')
    params = init_params(model_options)

    optimizer_params = {}
    # prepare parameters
    if reload_ and os.path.exists(loadfrom):
        logging.info('Reloading model parameters')
        params = load_params(loadfrom, params, ignore_list)
        if reload_training_progress:
            logging.info('Reloading optimizer parameters')
            try:
                logging.info('trying to load optimizer params from {0} or {1}'.format(loadfrom + '.gradinfo', loadfrom + '.gradinfo.npz'))
                optimizer_params = load_optimizer_params(loadfrom + '.gradinfo', optimizer)
            except IOError:
                logging.warning('{0}(.npz) not found. Trying to load optimizer params from {1}(.npz)'.format(loadfrom + '.gradinfo', loadfrom))
                optimizer_params = load_optimizer_params(loadfrom, optimizer)
    elif prior_model:
        logging.info('Initializing model parameters from prior')
        params = load_params(prior_model, params)

    # load prior model if specified
    if prior_model:
        logging.info('Loading prior model parameters')
        params = load_params(prior_model, params, with_prefix='prior_')

    tparams = init_theano_params(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        cost = \
        build_model(tparams, model_options)

    inps = [x, x_mask, y, y_mask]

    if validFreq or sampleFreq:
        logging.info('Building sampler')
        f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)


    # before any regularizer
    logging.info('Building f_log_probs...')
    f_log_probs = theano.function(inps, cost, profile=profile)
    logging.info('Done')

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            if kk.startswith('prior_'):
                continue
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # apply L2 regularisation to loaded model (map training)
    if map_decay_c > 0:
        map_decay_c = theano.shared(numpy_floatX(map_decay_c), name="map_decay_c")
        weight_map_decay = 0.
        for kk, vv in tparams.iteritems():
            if kk.startswith('prior_'):
                continue
            init_value = tparams['prior_' + kk]
            weight_map_decay += ((vv -init_value) ** 2).sum()
        weight_map_decay *= map_decay_c
        cost += weight_map_decay

    updated_params = OrderedDict(tparams)

    # don't update prior model parameters
    if prior_model:
        updated_params = OrderedDict([(key,value) for (key,value) in updated_params.iteritems() if not key.startswith('prior_')])

    logging.info('Computing gradient...')
    grads = tensor.grad(cost, wrt=itemlist(updated_params))
    logging.info('Done')

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')

    logging.info('Building optimizers...')
    f_update, optimizer_tparams = eval(optimizer)(lr, updated_params,
                                                                 grads, inps, cost,
                                                                 profile=profile,
                                                                 optimizer_params=optimizer_params)
    logging.info('Done')

    logging.info('Total compilation time: {0:.1f}s'.format(time.time() - comp_start))

    if validFreq == -1 or saveFreq == -1 or sampleFreq == -1:
        logging.info('Computing number of training batches')
        num_batches = len(train)
        logging.info('There are {} batches in the train set'.format(num_batches))

        if validFreq == -1:
            validFreq = num_batches
        if saveFreq == -1:
            saveFreq = num_batches
        if sampleFreq == -1:
            sampleFreq = num_batches

    logging.info('Optimization')

    #save model options
    json.dump(model_options, open('%s.json' % saveto, 'wb'), indent=2)

    valid_err = None

    cost_sum = 0
    cost_batches = 0
    last_disp_samples = 0
    last_words = 0
    ud_start = time.time()
    # p_validation = None


    for training_progress.eidx in xrange(training_progress.eidx, max_epochs):
        n_samples = 0

        for x, y in train:
            training_progress.uidx += 1
            use_noise.set_value(1.)

            #ensure consistency in number of factors
            if len(x) and len(x[0]) and len(x[0][0]) != factors:
                logging.error('Mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(factors, len(x[0][0])))
                sys.exit(1)

            if model_options['objective'] in ['CE', 'RAML']:

                sample_weights = [1.0] * len(y)
                
                xlen = len(x)
                n_samples += xlen

                x, x_mask, y, y_mask, sample_weights = prepare_data(x, y, weights=sample_weights,
                                                                    maxlen=maxlen,
                                                                    n_factors=factors,
                                                                    n_words_src=n_words_src,
                                                                    n_words=n_words)

                if x is None:
                    logging.warning('Minibatch with zero sample under length %d' % maxlen)
                    training_progress.uidx -= 1
                    continue
                
                cost_batches += 1
                last_disp_samples += xlen
                last_words += (numpy.sum(x_mask) + numpy.sum(y_mask))/2.0

                # compute cost, grads and update parameters
                cost = f_update(lrate, x, x_mask, y, y_mask)

                cost_sum += cost

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                logging.warning('NaN detected')
                return 1., 1., 1.

            # verbose
            if numpy.mod(training_progress.uidx, dispFreq) == 0:
                ud = time.time() - ud_start
                sps = last_disp_samples / float(ud)
                wps = last_words / float(ud)
                cost_avg = cost_sum / float(cost_batches)
                logging.info(
                    'Epoch {epoch} Update {update} Cost {cost} UD {ud} {sps} {wps}'.format(
                        epoch=training_progress.eidx,
                        update=training_progress.uidx,
                        cost=cost_avg,
                        ud=ud,
                        sps="{0:.2f} sents/s".format(sps),
                        wps="{0:.2f} words/s".format(wps)
                    )
                )
                ud_start = time.time()
                cost_batches = 0
                last_disp_samples = 0
                last_words = 0
                cost_sum = 0

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(training_progress.uidx, saveFreq) == 0:
                logging.info('Saving the best model...')
                if best_p is not None:
                    params = best_p
                    optimizer_params = best_opt_p
                else:
                    params = unzip_from_theano(tparams, excluding_prefix='prior_')
                    optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')

                save(params, optimizer_params, training_progress, saveto)
                logging.info('Done')

                # save with uidx
                if not overwrite:
                    logging.info('Saving the model at iteration {}...'.format(training_progress.uidx))
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], training_progress.uidx)

                    params = unzip_from_theano(tparams, excluding_prefix='prior_')
                    optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')
                    save(params, optimizer_params, training_progress, saveto_uidx)
                    logging.info('Done')


            # generate some samples with the model and display them
            if sampleFreq and numpy.mod(training_progress.uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[2])):
                    stochastic = True
                    x_current = x[:, :, jj][:, :, None]

                    # remove padding
                    x_current = x_current[:,:x_mask.astype('int64')[:, jj].sum(),:]

                    sample, score, sample_word_probs = gen_sample([f_init], [f_next],
                                               x_current,
                                               trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=True)
                    print 'Source ', jj, ': ',
                    for pos in range(x.shape[1]):
                        if x[0, pos, jj] == 0:
                            break
                        for factor in range(factors):
                            vv = x[factor, pos, jj]
                            if vv in worddicts_r[factor]:
                                sys.stdout.write(worddicts_r[factor][vv])
                            else:
                                sys.stdout.write('UNK')
                            if factor+1 < factors:
                                sys.stdout.write('|')
                            else:
                                sys.stdout.write(' ')
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if valid is not None and validFreq and numpy.mod(training_progress.uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                training_progress.history_errs.append(float(valid_err))

                if training_progress.uidx == 0 or valid_err <= numpy.array(training_progress.history_errs).min():
                    best_p = unzip_from_theano(tparams, excluding_prefix='prior_')
                    best_opt_p = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')
                    training_progress.bad_counter = 0
                if valid_err > numpy.array(training_progress.history_errs).min():
                    training_progress.bad_counter += 1
                    logging.info('Valid bad counter %s at %s' % (
                        training_progress.bad_counter, training_progress.uidx))
                    if training_progress.bad_counter > patience:

                        if training_progress.anneal_restarts_done < anneal_restarts:
                            logging.info('No progress on the validation set, annealing learning rate and resuming from best params.')
                            lrate *= anneal_decay
                            logging.info('Anneal learning rate to %s at %s' %(
                                lrate, training_progress.uidx
                            ))
                            decodeFreq *= \
                                anneal_decay ** training_progress.anneal_restarts_done
                            logging.info('Change decode frequent to %s' % decodeFreq)
                            training_progress.anneal_restarts_done += 1
                            training_progress.bad_counter = 0

                            # reload best parameters
                            if best_p is not None:
                                logging.info('Reloading best params...')
                                zip_to_theano(best_p, tparams)
                                logging.info('Done')

                            # reset optimizer parameters
                            logging.info('Reset model optimizer...')
                            for item in optimizer_tparams.values():

                                item.set_value(numpy.array(item.get_value()) * 0.)
                            logging.info('Done')

                        # stop
                        elif training_progress.bad_counter > (patience * 1000):
                            logging.info('Valid {}'.format(valid_err))
                            logging.info('Early Stop!')
                            training_progress.estop = True
                            break

                logging.info('Valid {}'.format(valid_err))

            # finish after this many updates
            if training_progress.uidx >= finish_after:
                logging.info('Finishing after %d iterations!' % training_progress.uidx)
                training_progress.estop = True
                break

        logging.info('Seen %d samples' % n_samples)

        if training_progress.estop:
            break

    if best_p is not None:
        zip_to_theano(best_p, tparams)
        zip_to_theano(best_opt_p, optimizer_tparams)

    if valid is not None:
        use_noise.set_value(0.)
        valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
        valid_err = valid_errs.mean()

        logging.info('Valid {}'.format(valid_err))

    if best_p is not None:
        params = copy.copy(best_p)
        optimizer_params = copy.copy(best_opt_p)

    else:
        params = unzip_from_theano(tparams, excluding_prefix='prior_')
        optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')

    save(params, optimizer_params, training_progress, saveto)

    return valid_err

