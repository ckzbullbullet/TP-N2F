import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
import collections
import math
import numpy as np
import subprocess
from tqdm import tqdm, trange
from pytorch_seq2seq.model import TPN2F
from sklearn.metrics import precision_recall_fscore_support as score
from queue import PriorityQueue
from lisp_exec import code_lisp
from lisp_exec import code_trace
from lisp_exec import code_types
from lisp_exec import data


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    temph = []
    tempr = []
    for h in hypothesis:
        temph.extend(h)
    for r in reference:
        tempr.extend(r)
    hypothesis = temph
    reference = tempr
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)



def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)



def compute_metrics(all_stats):
    tests_num = 0
    programs_num = 0
    bleu_acc = 0.0
    correct_program_acc = 0
    # Almost correct programs are those that were executed on more than one test and passed at least 50% tests.
    almost_correct_program_acc = 0
    exact_code_match_acc = 0
    syntax_error_acc = 0
    runtime_exception_acc = 0
    other_exception_acc = 0
    for stats in all_stats:
        tests_num += stats['tests-executed']
        programs_num += 1
        correct_program_acc += stats['correct-program']
        if (stats['correct-program'] != 0 or
                stats['tests-executed'] > 1 and stats['tests-passed']/stats['tests-executed'] >= 0.5):
            almost_correct_program_acc += 1
        exact_code_match_acc += stats['exact-code-match']
        syntax_error_acc += stats['syntax-error']
        runtime_exception_acc += stats['runtime-exception']
        other_exception_acc += len(stats['exceptions'])

    return {'accuracy': (correct_program_acc/programs_num) if programs_num else 0.0,
            '50p_accuracy': (almost_correct_program_acc/programs_num) if programs_num else 0.0,
            'exact_match_accuracy': (exact_code_match_acc/programs_num) if programs_num else 0.0,
            'syntax_error_freq': (syntax_error_acc/tests_num) if tests_num else 0.0,
            'runtime_exception_freq': (runtime_exception_acc/tests_num) if tests_num else 0.0,
            'other_exception_freq': (other_exception_acc/tests_num) if tests_num else 0.0,
            'programs_num': programs_num,
            'tests_num': tests_num,
            'correct_program_num': correct_program_acc,
            'almost_correct_program_num': almost_correct_program_acc,
            'exact_code_match_num': exact_code_match_acc,
            }



def model_eval(model, eval_dataloader, eval_tests, eval_args, src_dict, trg_dict, max_seq_length, device, role_analysis=False):
    preds = []
    ground_truths = []
    output_pairs = []
    all_stats = []
    input_F = []
    input_R = []
    output_O=[]

    for input_src_ids, output_src_ids, input_trg_ids, output_trg_ids, exp_ids in eval_dataloader:

        input_trg_pred = Variable(torch.LongTensor([[trg_dict.arg2index['<s>'],trg_dict.arg2index['<s>'],trg_dict.arg2index['<s>'],trg_dict.arg2index['<s>']] for i in range(input_trg_ids.size(0))])).to(device)
        input_trg_pred = input_trg_pred.unsqueeze(1)

        with torch.no_grad():
            if role_analysis:
                input_trg_pred, aFs, aRs, o_vec = decode_minibatch(model,input_src_ids,input_trg_pred,max_seq_length,trg_dict, device, role_analysis=True)
            else:
                input_trg_pred = decode_minibatch(model,input_src_ids,input_trg_pred,max_seq_length,trg_dict, device)


            input_lines_trg = input_trg_pred.cpu().numpy()
            input_lines_trg = [
            [[trg_dict.index2opt[x[0]],trg_dict.index2arg[x[1]],trg_dict.index2arg[x[2]],trg_dict.index2arg[x[3]]] for x in line]
            for line in input_lines_trg
            ]

            output_trg_ids = output_trg_ids.cpu().numpy()
            output_trg_ids = [
            [[trg_dict.index2opt[x[0]],trg_dict.index2arg[x[1]],trg_dict.index2arg[x[2]],trg_dict.index2arg[x[3]]] for x in line]
            for line in output_trg_ids
            ]


            if (type(model) == TPN2F) and role_analysis:
                aF_ids = [
                [[i for i in x] for x in line] 
                for line in aFs
                ]

                aR_ids = [
                [[i for i in x] for x in line] 
                for line in aRs
                ]
            else:
                aF_ids = [
                1
                for line in input_lines_trg
                ]

                aR_ids = [
                1
                for line in input_lines_trg
                ]

            if type(model) == TPN2F and role_analysis:
                o_vecs = [
                [[i for i in x] for x in line] 
                for line in o_vec
                ]
            else:
                o_vecs = [
                [1,1,1,1]
                for line in input_lines_trg
                ]

            exp_ids = list(exp_ids.cpu().numpy())


            for sentence_pred, sentence_real, sentence_real_src, exp_id, sin_F, sin_R, sin_O in zip(input_lines_trg, output_trg_ids, output_src_ids, exp_ids, aF_ids, aR_ids, o_vecs):
                
                if ['</s>','</s>','</s>','</s>'] in sentence_pred:
                    index = sentence_pred.index(['</s>','</s>','</s>','</s>'])
                    sin_O = sin_O[:index]
                    if sentence_pred[0]==['<s>','<s>','<s>','<s>']:
                        sentence_pred=sentence_pred[1:index]
                        sin_O = sin_O[1:]
                    else:
                        sentence_pred=sentence_pred[:index]
                else:
                    index = len(sentence_pred)
                    if sentence_pred[0]==['<s>','<s>','<s>','<s>']:
                        sentence_pred=sentence_pred[1:index+1]
                        sin_O = sin_O[1:index+1]
                    else:
                        sentence_pred=sentence_pred[:index+1]
                preds.append(sentence_pred)
                output_O.append(sin_O)


                if ['</s>','</s>','</s>','</s>'] in sentence_real:
                    index = sentence_real.index(['</s>','</s>','</s>','</s>'])
                    sentence_real=sentence_real[:index]
                else:
                    index = len(sentence_real)
                    sentence_real=sentence_real[:index+1]

                ground_truths.append(sentence_real)

                if '</s>' in sentence_real_src:
                    index = sentence_real_src.index('</s>')
                    if role_analysis:
                        sentence_input_RF = sentence_real_src[1:-1]
                        sin_F = sin_F[:index]
                        sin_R = sin_R[:index]
                else:
                    index = len(sentence_real_src)+1
                    if role_analysis:
                        sentence_input_RF = sentence_real_src[1:]
                        sin_F = sin_F[:index+1]
                        sin_R = sin_R[:index+1]

                sentence_real_src = ['<s>'] + sentence_real_src[:index+1]

                output_pairs.append((sentence_real_src, sentence_pred))

                short_tree_real = parse_program2tree(sentence_real)
                short_tree_pred = parse_program2tree(sentence_pred)
                exp_tests = eval_tests[exp_id]
                exp_args = eval_args[exp_id]
                lisp_executor = LispExecutor()
                stats = evaluate_code(short_tree_pred, exp_args, exp_tests, lisp_executor)
                stats['exact-code-match'] = int(short_tree_real == short_tree_pred)
                stats['correct-program'] = int(stats['tests-executed'] == stats['tests-passed'])

                all_stats.append(stats)
                if role_analysis:
                    input_F.append((sentence_input_RF, sin_F))
                    input_R.append((sentence_input_RF, sin_R))

    result_report = compute_metrics(all_stats)

    if role_analysis:
        return get_bleu(preds,ground_truths), output_pairs, result_report, input_F, input_R, output_O
    else:
        return get_bleu(preds,ground_truths), output_pairs, result_report


def parse_program2tree(prog):
    command_dct = {}
    for cmd_ind, cmd in enumerate(prog):
        cmd_name = "#"+str(cmd_ind)
        short_tree = []
        for arg in cmd:
            if arg[0]=='#':
                if arg in command_dct:
                    short_tree.append(command_dct[arg])
                else:
                    short_tree.append("<unk>")
            elif arg!="<argpad>" and arg!="<s>" and arg!="</s>" and arg!="<pad>":
                short_tree.append(arg)
        command_dct[cmd_name] = short_tree
    return short_tree


def decode_minibatch(
    model,
    input_lines_src,
    input_lines_trg,
    max_seq_length, 
    trg_dict, 
    device,
    role_analysis = False
):
    for i in range(max_seq_length):
        if type(model) == TPN2F:
            if role_analysis:
                decoder_logit_o,decoder_logit_a1,decoder_logit_a2,decoder_logit_a3, aFs, aRs, output_o = model(input_lines_src, input_lines_trg, rela_analysis=True)
            else:
                decoder_logit_o,decoder_logit_a1,decoder_logit_a2,decoder_logit_a3, aFs, aRs = model(input_lines_src, input_lines_trg)
        else:
            ecoder_logit_o,decoder_logit_a1,decoder_logit_a2,decoder_logit_a3= model(input_lines_src, input_lines_trg)

        word_probs_o = model.decode(decoder_logit_o,'opt')
        word_probs_a1 = model.decode(decoder_logit_a1,'arg')
        word_probs_a2 = model.decode(decoder_logit_a2,'arg')
        word_probs_a3 = model.decode(decoder_logit_a3,'arg')

        decoder_argmax_o = word_probs_o.data.cpu().numpy().argmax(axis=-1)
        decoder_argmax_a1 = word_probs_a1.data.cpu().numpy().argmax(axis=-1)
        decoder_argmax_a2 = word_probs_a2.data.cpu().numpy().argmax(axis=-1)
        decoder_argmax_a3 = word_probs_a3.data.cpu().numpy().argmax(axis=-1)

        next_preds_o = Variable(
            torch.from_numpy(decoder_argmax_o[:, -1])
        ).unsqueeze(-1).to(device)
        next_preds_a1 = Variable(
            torch.from_numpy(decoder_argmax_a1[:, -1])
        ).unsqueeze(-1).to(device)
        next_preds_a2 = Variable(
            torch.from_numpy(decoder_argmax_a2[:, -1])
        ).unsqueeze(-1).to(device)
        next_preds_a3 = Variable(
            torch.from_numpy(decoder_argmax_a3[:, -1])
        ).unsqueeze(-1).to(device)

        next_preds = torch.cat((next_preds_o,next_preds_a1,next_preds_a2,next_preds_a3),1)

        input_lines_trg = torch.cat((input_lines_trg, next_preds.unsqueeze(1)),1)


    if type(model) == TPN2F:
        if role_analysis:
            return input_lines_trg, aFs.cpu().numpy(), aRs.cpu().numpy(), output_o.cpu().numpy()
        else:
            return input_lines_trg
    else:
        return input_lines_trg

ExecutionResult = collections.namedtuple('ExecutionResult', ['result', 'trace'])

class LispExecutor(object):

    def __init__(self):
        self.lisp_units = code_lisp.load_lisp_units()

    def get_search_args(self, arguments):
        local_units = {k: v for k, v in self.lisp_units.items() if v.compute}
        for name, type_ in arguments.items():
            local_units[name] = (code_lisp.Unit(
                name=name, description=name,
                args=[], return_type=code_types.str_to_type(type_), compute=None, computed_return_type=None))
        return local_units

    def execute(self, code, arguments, inputs, record_trace=True):
        arguments = [(arg, code_types.str_to_type(tp))
                     for arg, tp in arguments.items()]
        if data.is_flat_code(code):
            try:
                code, _ = data.unflatten_code(code, 'lisp')
            except:
                return ExecutionResult(None, None)
        try:
            code_lisp.test_lisp_validity(
                self.lisp_units, code, dict(arguments),
                code_types.Type.T)
        except (ValueError, KeyError, TypeError):
            return ExecutionResult(None, None)
        t = code_trace.CodeTrace() if record_trace else None
        func = code_lisp.compile_func(
            self.lisp_units, 'main', code,
            arguments, code_types.Type.T,
            trace=t)
        try:
            result = func(*[inputs[arg] for arg, _ in arguments])
        except Exception:
            raise ExecutorRuntimeException()
        return ExecutionResult(result, t)

    def compare(self, gold, prediction):
        return gold == prediction

class ExecutorSyntaxException(Exception):
    pass


class ExecutorRuntimeException(Exception):
    pass

def evaluate_code(code, arguments, tests, executor_):
    stats = {'tests-executed': len(tests), 'tests-passed': 0,
             'result-none': 0, 'syntax-error': 0, 'runtime-exception': 0, 'exceptions': []}
    if not code:
        return stats
    for test in tests:
        try:
            execution_result = executor_.execute(code, arguments, test['input'])
        except ExecutorSyntaxException as e:
            stats['syntax-error'] += 1
            stats['exceptions'].append(str(e))
            continue
        except ExecutorRuntimeException as e:
            stats['runtime-exception'] += 1
            stats['exceptions'].append(str(e))
            continue
        except Exception as e:
            #print("Exception: %s" % e)
            #traceback.print_exc()
            #print(code, arguments, test['input'])
            stats['exceptions'].append(str(e))
            continue
        if execution_result.result is None:
            stats['result-none'] += 1
        if executor_.compare(test['output'], execution_result.result):
            stats['tests-passed'] += 1
    return stats
