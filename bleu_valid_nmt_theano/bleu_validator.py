import os
import sys
import subprocess, numpy

from batch_beam_search import generate_sentences
from util import load_dict

def prepare_batches(seqs_x):
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

class BleuValidator:

    def __init__(self, options, **kwargs):
        self.src_dict = load_dict(options['dictionaries'][0])
        self.trg_dict = load_dict(options['dictionaries'][1])
        self.tmp_dir = kwargs['tmp_dir']
        self.translate_script = kwargs['translate_script']
        self.bleu_script = kwargs['bleu_script']
        self.valid_src = kwargs['bleuvalid_src']
        self.valid_trg = kwargs['bleuvalid_trg']
        self.n_words_src = options['n_words_src']
        self.batch_size = 16
        self.batches = self.prepare_data()
        os.system('mkdir -p %s' % self.tmp_dir)
        self.check_script() # check bleu script
        self.trg_idict = dict()
        for k, v in self.trg_dict.iteritems():
            self.trg_idict[v] = k

    def to_bleu_cmd(self, trans_file):
        #TODO according to the specific bleu script
        #TODO here is the multi-bleu.pl version
        return '%s %s < %s' % (self.bleu_script, self.valid_trg, trans_file)

    @staticmethod
    def parse_bleu_result(bleu_result):
        '''
        parse bleu result string
        :param bleu_result:
        multi-bleu.perl example:
        BLEU = 33.55, 71.9/43.3/26.1/15.7 (BP=0.998, ratio=0.998, hyp_len=26225, ref_len=26289)
        :return: float(33.55) or -1 for error
        '''
        #TODO according to the specific bleu script
        #TODO here is the multi-bleu.pl version
        bleu_result = bleu_result.strip()
        if bleu_result == '':
            return -1.
        try:
            bleu = float(bleu_result[7:bleu_result.index(',')])
        except ValueError:
            bleu = -1.
        return bleu

    def check_script(self):
        if not os.path.exists(self.bleu_script):
            print 'bleu script not exists: %s' % self.bleu_script
            sys.exit(0)
        if not os.path.exists(self.valid_src):
            print 'valid src file not exists: %s' % self.valid_src
            sys.exit(0)

        if os.path.exists(self.valid_trg):
            cmd = self.to_bleu_cmd(self.valid_trg)
        elif os.path.exists(self.valid_trg + str(0)):
            cmd = self.to_bleu_cmd(self.valid_trg + str(0))
        else:
            print 'valid trg file not exists: %s or %s0' % (self.valid_trg, self.valid_trg)
            sys.exit(0)

        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        popen.wait()
        bleu = BleuValidator.parse_bleu_result(''.join(popen.stdout.readlines()).strip())
        if bleu == -1:
            print 'Fail to run script: %s. Please CHECK' % self.bleu_script
            # sys.exit(0)
        else:
            print 'Successfully test bleu script: %s' % self.bleu_script

    def prepare_data(self):
        sources = []
        with open(self.valid_src, 'r') as f:
            for idx, line in enumerate(f):
                words = line.strip().split()
                if len(words) == 0:
                    continue
                x = map(lambda w: self.src_dict[w] if w in self.src_dict else 1, words)
                x = map(lambda ii: ii if ii < self.n_words_src else 1, x)
                sources.append(x)
        batches = []
        for i in range(len(sources) / self.batch_size):
            batches.append(prepare_batches(sources[i * self.batch_size: (i + 1) * self.batch_size]))
        if (i + 1) *  self.batch_size < len(sources):
            batches.append(prepare_batches(sources[(i + 1) *  self.batch_size:]))
        return batches

    def decode(self, device, trans_saveto, model_file, model_options):
        # TODO python translate.py -n xxx xxx xxx
        cmd = "THEANO_FLAGS='device=%s,floatX=float32' python %s --model %s --options %s --input %s --output %s --source_dic %s --target_dic %s " \
            % (device,
               self.translate_script,
               model_file,
               model_options,
               self.valid_src,
               os.path.join(self.tmp_dir, trans_saveto),
               self.src_dict,
               self.trg_dict)
        print 'running: %s' % cmd
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    def batch_decode(self, f_batch_beam_sampler, trans_saveto, model_options):
        final_sentences = []
        for batch in self.batches:
            final_word_ids, final_beam_ids, final_beam_scores = f_batch_beam_sampler(*batch)
            final_sentences += generate_sentences(final_word_ids, final_beam_ids,
                                                  final_beam_scores, self.trg_idict)
        with open(trans_saveto, 'w') as fout:
            for sentence in final_sentences:
                fout.write(sentence)


    def test_bleu(self, trans_file):
        popen = subprocess.Popen(self.to_bleu_cmd(os.path.join(self.tmp_dir, trans_file)),
                                 stdout=subprocess.PIPE, shell=True)
        popen.wait()
        bleu = BleuValidator.parse_bleu_result(''.join(popen.stdout.readlines()).strip())
        if bleu == -1:
            print 'Fail to run script: %s, for testing trans file: %s' % (self.bleu_script, trans_file)
            # sys.exit(0)
        return bleu

