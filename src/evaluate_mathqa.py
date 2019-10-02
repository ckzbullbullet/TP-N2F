import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
import math
import numpy as np
import subprocess
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support as score
from queue import PriorityQueue


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
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



def model_eval(model, eval_dataloader, src_dict, trg_dict, max_seq_length, device, args_binary_rela=False, rela_analysis=False):
    preds = []
    ground_truths = []
    prog_accuracy = []
    output_pairs = []
    input_F = []
    input_R = []
    output_O=[]
    print("binary : " + str(args_binary_rela))
    for input_src_ids, output_src_ids, input_trg_ids, output_trg_ids in eval_dataloader:
        if args_binary_rela:
            input_trg_pred = Variable(torch.LongTensor([[trg_dict.arg2index['<s>'],trg_dict.arg2index['<s>'],trg_dict.arg2index['<s>']] for i in range(input_trg_ids.size(0))])).to(device)
        else:
            input_trg_pred = Variable(torch.LongTensor([[trg_dict.arg2index['<s>'],trg_dict.arg2index['<s>'],trg_dict.arg2index['<s>'],trg_dict.arg2index['<s>']] for i in range(input_trg_ids.size(0))])).to(device)
        input_trg_pred = input_trg_pred.unsqueeze(1)

        if not args_binary_rela:
            with torch.no_grad():
                if rela_analysis:
                    input_trg_pred, aFs, aRs, o_vec = decode_minibatch(model,input_src_ids,input_trg_pred,max_seq_length,trg_dict, device, args_binary_rela,rela_analysis=True)
                else:
                    input_trg_pred, aFs, aRs= decode_minibatch(model,input_src_ids,input_trg_pred,max_seq_length,trg_dict, device, args_binary_rela)

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

                output_src_ids = output_src_ids.cpu().numpy()
                output_src_ids = [
                [src_dict.index2src[x] for x in line] 
                for line in output_src_ids
                ]

                aF_ids = [
                [[i for i in x] for x in line] 
                for line in aFs
                ]

                aR_ids = [
                [[i for i in x] for x in line] 
                for line in aRs
                ]


                if rela_analysis:
                    o_vecs = [
                    [[i for i in x] for x in line] 
                    for line in o_vec
                    ]
                else:
                    o_vecs = [
                    [1,1,1,1]
                    for line in input_lines_trg
                    ]

                for sentence_pred, sentence_real, sentence_real_src,sin_F, sin_R,sin_O in zip(input_lines_trg,output_trg_ids,output_src_ids,aF_ids,aR_ids, o_vecs):
                    
                    if ['</s>','</s>','</s>','</s>'] in sentence_pred:
                        index = sentence_pred.index(['</s>','</s>','</s>','</s>'])
                        sin_O = sin_O[:index]
                        if sentence_pred[0]==['<s>','<s>','<s>','<s>']:
                            sentence_pred=['<s>']+[val for sublist in sentence_pred[1:index] for val in sublist]+['</s>']
                            sin_O = sin_O[1:]
                        else:
                            sentence_pred=['<s>']+[val for sublist in sentence_pred[:index] for val in sublist]+['</s>']
                    else:
                        index = len(sentence_pred)
                        if sentence_pred[0]==['<s>','<s>','<s>','<s>']:
                            sentence_pred=['<s>']+[val for sublist in sentence_pred[1:index+1] for val in sublist]
                            sin_O = sin_O[1:]
                        else:
                            sentence_pred=['<s>']+[val for sublist in sentence_pred[:index+1] for val in sublist]
                    preds.append(sentence_pred)
                    output_O.append(sin_O)

                    if ['</s>','</s>','</s>','</s>'] in sentence_real:
                        index = sentence_real.index(['</s>','</s>','</s>','</s>'])
                        sentence_real=['<s>']+[val for sublist in sentence_real[:index] for val in sublist]+['</s>']
                    else:
                        index = len(sentence_real)
                        sentence_real=['<s>']+[val for sublist in sentence_real[:index+1] for val in sublist]+['</s>']

                    ground_truths.append(sentence_real)

                    if '</s>' in sentence_real_src:
                        index = sentence_real_src.index('</s>')
                        sentence_real_src = ['<s>'] + sentence_real_src[:index+1]
                        sentence_input_RF = sentence_real_src[1:-1]
                        sin_F = sin_F[:index]
                        sin_R = sin_R[:index]
                    else:
                        index = len(sentence_real_src)+1
                        sentence_real_src = ['<s>'] + sentence_real_src[:index+1]
                        sentence_input_RF = sentence_real_src[1:]
                        sin_F = sin_F[:index+1]
                        sin_R = sin_R[:index+1]


                    if sentence_real == sentence_pred:
                        prog_accuracy.append(1)
                    else:
                        prog_accuracy.append(0)

                    output_pairs.append((sentence_real_src, sentence_pred))
                    input_F.append((sentence_input_RF, sin_F))
                    input_R.append((sentence_input_RF, sin_R))
        else:
            with torch.no_grad():
                if rela_analysis:
                    input_trg_pred, aFs, aRs, o_vec = decode_minibatch(model,input_src_ids,input_trg_pred,max_seq_length,trg_dict, device, args_binary_rela,rela_analysis=rela_analysis)
                else:
                    input_trg_pred, aFs, aRs= decode_minibatch(model,input_src_ids,input_trg_pred,max_seq_length,trg_dict, device, args_binary_rela)
                
                input_lines_trg = input_trg_pred.cpu().numpy()
                input_lines_trg = [
                [[trg_dict.index2opt[x[0]],trg_dict.index2arg[x[1]],trg_dict.index2arg[x[2]]] for x in line]
                for line in input_lines_trg
                ]

                output_trg_ids = output_trg_ids.cpu().numpy()
                output_trg_ids = [
                [[trg_dict.index2opt[x[0]],trg_dict.index2arg[x[1]],trg_dict.index2arg[x[2]]] for x in line]
                for line in output_trg_ids
                ]

                output_src_ids = output_src_ids.cpu().numpy()
                output_src_ids = [
                [src_dict.index2src[x] for x in line] 
                for line in output_src_ids
                ]

                aF_ids = [
                [[i for i in x] for x in line] 
                for line in aFs
                ]

                aR_ids = [
                [[i for i in x] for x in line] 
                for line in aRs
                ]


                if rela_analysis:
                    o_vecs = [
                    [[i for i in x] for x in line] 
                    for line in o_vec
                    ]
                else:
                    o_vecs = [
                    [1,1,1,1]
                    for line in input_lines_trg
                    ]

                for sentence_pred, sentence_real, sentence_real_src,sin_F, sin_R, sin_O in zip(input_lines_trg,output_trg_ids,output_src_ids,aF_ids,aR_ids,o_vecs):
                    
                    if ['</s>','</s>','</s>'] in sentence_pred:
                        index = sentence_pred.index(['</s>','</s>','</s>'])
                        sin_O = sin_O[:index]
                        if sentence_pred[0]==['<s>','<s>','<s>']:
                            sin_O = sin_O[1:]
                            sentence_pred=['<s>']+[val for sublist in sentence_pred[1:index] for val in sublist]+['</s>']
                        else:
                            sentence_pred=['<s>']+[val for sublist in sentence_pred[:index] for val in sublist]+['</s>']
                    else:
                        index = len(sentence_pred)
                        if sentence_pred[0]==['<s>','<s>','<s>']:
                            sin_O = sin_O[1:]
                            sentence_pred=['<s>']+[val for sublist in sentence_pred[1:index+1] for val in sublist]
                        else:
                            sentence_pred=['<s>']+[val for sublist in sentence_pred[:index+1] for val in sublist]
                    preds.append(sentence_pred)
                    output_O.append(sin_O)


                    if ['</s>','</s>','</s>'] in sentence_real:
                        index = sentence_real.index(['</s>','</s>','</s>'])
                        sentence_real=['<s>']+[val for sublist in sentence_real[:index] for val in sublist]+['</s>']
                    else:
                        index = len(sentence_real)
                        sentence_real=['<s>']+[val for sublist in sentence_real[:index+1] for val in sublist]+['</s>']

                    ground_truths.append(sentence_real)

                    if '</s>' in sentence_real_src:
                        index = sentence_real_src.index('</s>')
                        sentence_real_src = ['<s>'] + sentence_real_src[:index+1]
                        sentence_input_RF = sentence_real_src[1:-1]
                        sin_F = sin_F[:index]
                        sin_R = sin_R[:index]
                    else:
                        index = len(sentence_real_src)+1
                        sentence_real_src = ['<s>'] + sentence_real_src[:index+1]
                        sentence_input_RF = sentence_real_src[1:]
                        sin_F = sin_F[:index+1]
                        sin_R = sin_R[:index+1]
                    if sentence_real == sentence_pred:
                        prog_accuracy.append(1)
                    else:
                        prog_accuracy.append(0)

                    output_pairs.append((sentence_real_src, sentence_pred))
                    input_F.append((sentence_input_RF, sin_F))
                    input_R.append((sentence_input_RF, sin_R))
                
    if rela_analysis:
        return get_bleu(preds,ground_truths), sum(prog_accuracy)/float(len(prog_accuracy)), output_pairs,input_F,input_R, output_O
    else:
        return get_bleu(preds,ground_truths), sum(prog_accuracy)/float(len(prog_accuracy)), output_pairs,input_F,input_R 





def decode_minibatch(
    model,
    input_lines_src,
    input_lines_trg,
    max_seq_length, 
    trg_dict, 
    device,
    args_binary_rela = False,
    rela_analysis=False
):
    for i in range(max_seq_length):
        if not args_binary_rela:
            if rela_analysis:
                decoder_logit_o,decoder_logit_a1,decoder_logit_a2,decoder_logit_a3, output_o, aFs, aRs = model(input_lines_src, input_lines_trg, rela_analysis=True)
            else:
                decoder_logit_o,decoder_logit_a1,decoder_logit_a2,decoder_logit_a3, aFs, aRs = model(input_lines_src, input_lines_trg)
        else:
            if rela_analysis:
                decoder_logit_o,decoder_logit_a1,decoder_logit_a2, output_o, aFs, aRs= model(input_lines_src, input_lines_trg, rela_analysis=True)
            else:
                decoder_logit_o,decoder_logit_a1,decoder_logit_a2,aFs, aRs= model(input_lines_src, input_lines_trg)

        word_probs_o = model.decode(decoder_logit_o,'opt')
        word_probs_a1 = model.decode(decoder_logit_a1,'arg')
        word_probs_a2 = model.decode(decoder_logit_a2,'arg')
        if not args_binary_rela:
            word_probs_a3 = model.decode(decoder_logit_a3,'arg')

        decoder_argmax_o = word_probs_o.data.cpu().numpy().argmax(axis=-1)
        decoder_argmax_a1 = word_probs_a1.data.cpu().numpy().argmax(axis=-1)
        decoder_argmax_a2 = word_probs_a2.data.cpu().numpy().argmax(axis=-1)
        if not args_binary_rela:
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
        if not args_binary_rela:
            next_preds_a3 = Variable(
                torch.from_numpy(decoder_argmax_a3[:, -1])
            ).unsqueeze(-1).to(device)

        if rela_analysis:
            if i==0:
                output_o_vectors = output_o[:,-1,:].unsqueeze(1)
            else:
                output_o_vectors = torch.cat((output_o_vectors,output_o[:,-1,:].unsqueeze(1)), 1)


        if not args_binary_rela:
            next_preds = torch.cat((next_preds_o,next_preds_a1,next_preds_a2,next_preds_a3),1)
        else:
            next_preds = torch.cat((next_preds_o,next_preds_a1,next_preds_a2),1)

        input_lines_trg = torch.cat((input_lines_trg, next_preds.unsqueeze(1)),1)


    if rela_analysis:
        return input_lines_trg, aFs.cpu().numpy(), aRs.cpu().numpy(), output_o_vectors.cpu().numpy()
    else:
        return input_lines_trg, aFs.cpu().numpy(), aRs.cpu().numpy()
