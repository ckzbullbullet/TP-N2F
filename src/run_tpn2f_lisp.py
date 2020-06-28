
from __future__ import absolute_import

import argparse
import json
import logging
import os
import random
import numpy as np
from tqdm import tqdm, trange
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable

from model import TPN2F
import evaluate_lisp as evaluate


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class LispExample(object):
    def __init__(self,
                 text,
                 code_tree,
                 code_sequence,
                 code_commands_pad,
                 code_commands,
                 tests,
                 args):
        self.text = text
        self.code_tree = code_tree
        self.code_sequence = code_sequence
        self.code_commands_pad = code_commands_pad
        self.code_commands = code_commands
        self.tests = tests
        self.args = args


class SourceDict(object):
    def __init__(self, filepath):
        self.vocab = {}

        self.src2index = {}

        self.index2src = {}

        self.construct_vocab(filepath)
        self.n_srcs = len(self.src2index)

    def construct_vocab(self, filepath):
        with open(filepath, 'r') as f:
            vocabs = f.read().strip().split("\n")
        for vocab in vocabs:
            vocab = vocab.split(" : ")
            vind = int(vocab[0])
            word = vocab[1]
            self.src2index[word] = vind
            self.index2src[vind] = word



class TargetDict(object):
    def __init__(self, opt_path, arg_path):
        self.opt2index = {}
        self.index2opt = {}
        self.arg2index = {}
        self.index2arg = {}
        self.create_vocb(opt_path, arg_path)
        self.n_opts = len(self.index2opt)
        self.n_args = len(self.index2arg)

    def create_vocb(self, opt_path, arg_path):
        with open(opt_path,  'r') as f:
            opt_vocabs = f.read().strip().split("\n")
        with open(arg_path, 'r') as f:
            arg_vocabs = f.read().strip().split("\n")
        for vocab in opt_vocabs:
            vocab = vocab.split(" : ")
            vind = int(vocab[0])
            word = vocab[1]
            self.opt2index[word] = vind
            self.index2opt[vind] = word
        for vocab in arg_vocabs:
            vocab = vocab.split(" : ")
            vind = int(vocab[0])
            word = vocab[1]
            self.arg2index[word] = vind
            self.index2arg[vind] = word

 

class InputFeatures(object):
    def __init__(self,
                 input_src_ids,
                 output_src_ids,
                 input_trg_ids,
                 output_trg_ids,
                 tests=None,
                 exp_ind=None,
                 args=None):
        self.input_src_ids = input_src_ids
        self.output_src_ids = output_src_ids
        self.input_trg_ids = input_trg_ids
        self.output_trg_ids = output_trg_ids
        self.tests = tests
        self.exp_ind = exp_ind
        self.args = args


def read_lisp_examples(input_file):
    with open(input_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    examples = [
        LispExample(
            d['text'],
            d['code_tree'],
            d['code_sequence'],
            d['code_commands_pad'],
            d['code_commands'],
            d['tests'],
            d['args']
            ) for d_index, d in enumerate(data)
    ]
    return examples


def convert_examples_to_features(examples, max_seq_length, src_dict, trg_dict):
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = example.text
        target_tokens = example.code_commands_pad
        example_tests = example.tests
        example_args = example.args
        _truncate_seq_pair(context_tokens, max_seq_length-2)
        _truncate_seq_pair(target_tokens, max_seq_length-2)
        context_tokens = ['<s>'] + context_tokens + ['</s>']
        target_tokens = [['<s>','<s>','<s>','<s>']] + target_tokens + [['</s>','</s>','</s>','</s>']]

        input_src_ids = [src_dict.src2index[i] if i in src_dict.src2index else src_dict.src2index['<unk>'] for i in context_tokens[:-1]] + [src_dict.src2index['<pad>']]*(max_seq_length - len(context_tokens)+1)

        output_src_ids = [src_dict.src2index[i] if i in src_dict.src2index else src_dict.src2index['<unk>'] for i in context_tokens[1:]] + [src_dict.src2index['<pad>']]*(max_seq_length - len(context_tokens)+1)
        input_trg_ids = []
        output_trg_ids = []
        for x in target_tokens[:-1]:
            tempopt = trg_dict.opt2index[x[0]] if x[0] in trg_dict.opt2index else trg_dict.opt2index['<unk>']
            tempa1 = trg_dict.arg2index[x[1]] if x[1] in trg_dict.arg2index else trg_dict.arg2index['<unk>']
            tempa2 = trg_dict.arg2index[x[2]] if x[2] in trg_dict.arg2index else trg_dict.arg2index['<unk>']
            tempa3 = trg_dict.arg2index[x[3]] if x[3] in trg_dict.arg2index else trg_dict.arg2index['<unk>']
            input_trg_ids.append([tempopt, tempa1, tempa2, tempa3])
        for x in target_tokens[1:]:
            tempopt = trg_dict.opt2index[x[0]] if x[0] in trg_dict.opt2index else trg_dict.opt2index['<unk>']
            tempa1 = trg_dict.arg2index[x[1]] if x[1] in trg_dict.arg2index else trg_dict.arg2index['<unk>']
            tempa2 = trg_dict.arg2index[x[2]] if x[2] in trg_dict.arg2index else trg_dict.arg2index['<unk>']
            tempa3 = trg_dict.arg2index[x[3]] if x[3] in trg_dict.arg2index else trg_dict.arg2index['<unk>']
            output_trg_ids.append([tempopt, tempa1, tempa2, tempa3])
        input_trg_ids = input_trg_ids + [[trg_dict.opt2index['<pad>'],trg_dict.arg2index['<pad>'],trg_dict.arg2index['<pad>'],trg_dict.arg2index['<pad>']]]*(max_seq_length - len(target_tokens)+1)
        output_trg_ids = output_trg_ids + [[trg_dict.opt2index['<pad>'],trg_dict.arg2index['<pad>'],trg_dict.arg2index['<pad>'],trg_dict.arg2index['<pad>']]]*(max_seq_length - len(target_tokens)+1)

        features.append(
            InputFeatures(
                input_src_ids = input_src_ids,
                output_src_ids = output_src_ids,
                input_trg_ids = input_trg_ids,
                output_trg_ids = output_trg_ids, 
                tests = example_tests,
                exp_ind = example_index,
                args = example_args
            )
        )

    return features


def _truncate_seq_pair(tokens_a, max_length):

    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default=os.getenv('PT_OUTPUT_DIR', '/tmp'),
                        type=str,
                        help="The output directory where the model checkpoints will be written.")


    ## Other parameters
    parser.add_argument("--eval_model_file", default='', type=str, 
                        help="for evaluation, the model file stored")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--optimizer",
                        default='adam',
                        type=str,
                        help="adam/adadelta/sgd")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run evaluation.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=100.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--checkpoint_freq",
                        type=int,
                        default=-1,
                        help="model checkpoint each #n of epochs (save the model each #n of epochs)")

    parser.add_argument("--src_layer",
                        type=int,
                        default=1,
                        help="src_layer numbers in TP-N2F encoder")

    parser.add_argument("--trg_layer",
                        type=int,
                        default=1,
                        help="trg_layer numbers in TP-N2F decoder")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--nSymbols',
                        type=int, default=150,
                        help="number of fillers in TP-N2F encoder")

    parser.add_argument('--nRoles',
                        type=int, default=50,
                        help="number of roles in TP-N2F encoder")

    parser.add_argument('--dSymbols',
                        type=int, default=30,
                        help="dimension of the filler vector in TP-N2F encoder")

    parser.add_argument('--dRoles',
                        type=int, default=30,
                        help="dimension of the role vector in TP-N2F encoder")

    parser.add_argument('--temperature',
                        type=float, default=0.1,
                        help="temperature in TP-N2F encoder")

    parser.add_argument('--dOpts',
                        type=int, default=30,
                        help="dimension of operator vector in TP-N2F decoder")

    parser.add_argument('--dArgs',
                        type=int, default=20,
                        help="dimension of argument vector in TP-N2F decoder")

    parser.add_argument('--dPoss',
                        type=int, default=5,
                        help="dimension of position vector in TP-N2F decoder")

    parser.add_argument('--role_grad',
                        type=str, default="True",
                        help="whether position vector is trained or one-hot vectors")

    parser.add_argument('--attention',
                        type=str, default='dot',
                        help="attention type dot/tpr")

    parser.add_argument('--sum_T',
                        type=str, default="True",
                        help="Whether the output of TP-N2F encoder is the sum of all tensor product or the last one")

    parser.add_argument('--reason_T',
                        type=int, default=1,
                        help="number of layers of the TP-N2F reasoning MLP. Only support 1 or 2 layers")

    parser.add_argument('--bidirectional',
                        type=str, default="False",
                        help="bidirectional for encoder")

    parser.add_argument('--lr_decay',
                        type=str, default="True",
                        help="whether have learning rate decay")

    parser.add_argument('--clean_data',
                        type=int, default=0,
                        help="whether test on the full testing set or cleaned testing set")


    args = parser.parse_args()

    args_role_grad = True if args.role_grad=="True" else False
    args_sum_T = True if args.sum_T=="True" else False
    args_bidirectional = True if args.bidirectional=="True" else False
    args_lr_decay = True if args.lr_decay=="True" else False


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_filepath = os.path.join(args.data_dir, 'algolisp_train.json')
    dev_filepath = os.path.join(args.data_dir, 'algolisp_dev.json')
    if args.clean_data==1:
        test_filepath = os.path.join(args.data_dir, 'algolisp_test_clean.json')
    else:
        test_filepath = os.path.join(args.data_dir, 'algolisp_test.json')


    logger.info("***** Start Building target dictionary *****")
    trg_dict = TargetDict(os.path.join(args.data_dir, 'opt_vocab.txt'), os.path.join(args.data_dir, 'arg_vocab.txt'))
    logger.info("***** Start Building source dictionary *****")
    src_dict = SourceDict(os.path.join(args.data_dir, 'text_vocab.txt'))


    weight_mask_opt = torch.ones(trg_dict.n_opts).to(device)
    weight_mask_opt[trg_dict.opt2index['<pad>']] = 0
    weight_mask_arg = torch.ones(trg_dict.n_args).to(device)
    weight_mask_arg[trg_dict.arg2index['<pad>']] = 0
    loss_criterion_opt = nn.CrossEntropyLoss(weight=weight_mask_opt).to(device)
    loss_criterion_arg1 = nn.CrossEntropyLoss(weight=weight_mask_arg).to(device)
    loss_criterion_arg2 = nn.CrossEntropyLoss(weight=weight_mask_arg).to(device)
    loss_criterion_arg3 = nn.CrossEntropyLoss(weight=weight_mask_arg).to(device)

    if args.do_train:
        model_batch_size = args.train_batch_size
    else:
        model_batch_size = args.eval_batch_size

    logger.info("Prepare Model.")
    model = TPN2F(
    src_emb_dim=args.max_seq_length,
    trg_emb_dim=args.max_seq_length,
    src_vocab_size=src_dict.n_srcs,
    opt_vocab_size=trg_dict.n_opts,
    arg_vocab_size=trg_dict.n_args,
    attention_mode=args.attention,
    pad_token_src=src_dict.src2index['<pad>'],
    pad_token_opt=trg_dict.opt2index['<pad>'],
    pad_token_arg=trg_dict.arg2index['<pad>'],
    nlayers=args.src_layer,
    bidirectional=args_bidirectional,
    nSymbols=args.nSymbols, nRoles=args.nRoles, dSymbols=args.dSymbols, dRoles=args.dRoles,
    temperature=args.temperature, dOpts=args.dOpts, dArgs=args.dArgs, dPoss=args.dPoss,
    sum_T = args_sum_T, reason_T = args.reason_T, binary=False
    ).to(device)


    if not args.do_train and args.do_eval:
        try:
            logging.info("The experiment only for evalulation.")
            model.load_state_dict(torch.load(os.path.join(args.data_dir, args.eval_model_file)))
            model.eval()
        except:
            if args.no_cuda:
                checkpoint = torch.load(os.path.join(args.data_dir, args.eval_model_file),map_location='cpu')
            else:
                checkpoint = torch.load(os.path.join(args.data_dir, args.eval_model_file))

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    if n_gpu > 1 and args.do_train:
        model = torch.nn.DataParallel(model)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer  == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif args.optimizer  == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    if args_lr_decay:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

    if args.do_train:
        train_examples = read_lisp_examples(train_filepath)
        train_features = convert_examples_to_features(train_examples, args.max_seq_length, src_dict, trg_dict)
        all_input_src_ids = Variable(torch.tensor([f.input_src_ids for f in train_features],dtype=torch.long)).to(device)
        all_output_src_ids = Variable(torch.tensor([f.output_src_ids for f in train_features],dtype=torch.long)).to(device)
        all_input_trg_ids = Variable(torch.tensor([f.input_trg_ids for f in train_features],dtype=torch.long)).to(device)
        all_output_trg_ids = Variable(torch.tensor([f.output_trg_ids for f in train_features],dtype=torch.long)).to(device)
        train_data = TensorDataset(all_input_src_ids, all_output_src_ids, all_input_trg_ids, all_output_trg_ids)

        dev_examples = read_lisp_examples(dev_filepath)
        dev_features = convert_examples_to_features(dev_examples, args.max_seq_length, src_dict, trg_dict)
        all_input_src_ids_dev = Variable(torch.tensor([f.input_src_ids for f in dev_features],dtype=torch.long)).to(device)
        all_output_src_ids_dev = Variable(torch.tensor([f.output_src_ids for f in dev_features],dtype=torch.long)).to(device)
        all_input_trg_ids_dev = Variable(torch.tensor([f.input_trg_ids for f in dev_features],dtype=torch.long)).to(device)
        all_output_trg_ids_dev = Variable(torch.tensor([f.output_trg_ids for f in dev_features],dtype=torch.long)).to(device)
        dev_data = TensorDataset(all_input_src_ids_dev, all_output_src_ids_dev, all_input_trg_ids_dev, all_output_trg_ids_dev)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        dev_sampler = RandomSampler(dev_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.train_batch_size)
        train_dev_dataloader = {"train": train_dataloader, "dev": dev_dataloader}

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d for training", len(train_examples))
        logger.info("  Num examples = %d for validation", len(dev_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.num_train_epochs)

        #model.train()

        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            for phase in ['train', 'dev']:
                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)

                losses = []
                for step, batch in enumerate(train_dev_dataloader[phase]):
                    batch = tuple(t.to(device) for t in batch)
                    input_src_ids, output_src_ids, input_trg_ids, output_trg_ids= batch
                    
                    output_opt_ids = output_trg_ids[:,:,0]
                    output_a1_ids = output_trg_ids[:,:,1]
                    output_a2_ids = output_trg_ids[:,:,2]
                    output_a3_ids = output_trg_ids[:,:,3]
                    output_o, output_a1,output_a2, output_a3,aFs, aRs = model(input_src_ids, input_trg_ids)

                    optimizer.zero_grad()
                    
                    loss_opt = loss_criterion_opt(output_o.contiguous().view(-1, trg_dict.n_opts),output_opt_ids.view(-1))
                    loss_a1 = loss_criterion_arg1(output_a1.contiguous().view(-1, trg_dict.n_args),output_a1_ids.view(-1))
                    loss_a2 = loss_criterion_arg2(output_a2.contiguous().view(-1, trg_dict.n_args),output_a2_ids.view(-1))
                    loss_a3 = loss_criterion_arg3(output_a3.contiguous().view(-1, trg_dict.n_args),output_a3_ids.view(-1))
                    loss = loss_opt + loss_a1 + loss_a2 + loss_a3

                    losses.append(loss.item())
                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                if phase=='train' and args_lr_decay:
                    scheduler.step()

                logging.info('Epoch : %d Phase : %s Loss : %.5f' % (
                        ep, phase, np.mean(losses))
                )

            if args.checkpoint_freq > -1 and ep % args.checkpoint_freq == 0:
                logging.info('Reach model checkpoint at epoch: %d ' % (ep))
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_epoch' + str(ep) + '.model'))


        logging.info('Finish training epochs and saving the final model ...')
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.model'))
        if n_gpu > 1:
            checkpoint = model.state_dict()
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k[:7]=='module.':
                    name = k[7:]
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            model = TPN2F(
            src_emb_dim=args.max_seq_length,
            trg_emb_dim=args.max_seq_length,
            src_vocab_size=src_dict.n_srcs,
            opt_vocab_size=trg_dict.n_opts,
            arg_vocab_size=trg_dict.n_args,
            attention_mode=args.attention,
            pad_token_src=src_dict.src2index['<pad>'],
            pad_token_opt=trg_dict.opt2index['<pad>'],
            pad_token_arg=trg_dict.arg2index['<pad>'],
            nlayers=args.src_layer,
            bidirectional=args_bidirectional,
            nSymbols=args.nSymbols, nRoles=args.nRoles, dSymbols=args.dSymbols, dRoles=args.dRoles,
            temperature=args.temperature, dOpts=args.dOpts, dArgs=args.dArgs, dPoss=args.dPoss,
            sum_T = args_sum_T, reason_T = args.reason_T, binary=False
            ).to(device)
            model.load_state_dict(new_state_dict)


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = read_lisp_examples(test_filepath)
        eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, src_dict, trg_dict)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_src_ids = Variable(torch.tensor([f.input_src_ids for f in eval_features],dtype=torch.long)).to(device)
        all_output_src_ids = Variable(torch.tensor([f.output_src_ids for f in eval_features],dtype=torch.long)).to(device)
        all_input_trg_ids = Variable(torch.tensor([f.input_trg_ids for f in eval_features],dtype=torch.long)).to(device)
        all_output_trg_ids = Variable(torch.tensor([f.output_trg_ids for f in eval_features],dtype=torch.long)).to(device)
        all_exp_ids = Variable(torch.tensor([f.exp_ind for f in eval_features],dtype=torch.long))
        eval_tests = [f.tests for f in eval_features]
        eval_args = [f.args for f in eval_features]
        eval_data = TensorDataset(all_input_src_ids, all_output_src_ids, all_input_trg_ids, all_output_trg_ids,all_exp_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=model_batch_size)

        model.eval()
        bleu, result_pairs, result_report= evaluate.model_eval(model, eval_dataloader, eval_tests, eval_args, src_dict, trg_dict, args.max_seq_length,device, False)

        pred_file = os.path.join(args.output_dir, "test_predications.txt")
        with open(pred_file, 'w', encoding='utf8') as writer:
            for q_pair in result_pairs:
                question, pred_sen = q_pair
                p_str = []
                for command in pred_sen:
                    command = ["("] + command + [")"]
                    for p_char in command:
                        if p_char != '<unk>' and p_char != '</s>':
                            p_str.append(p_char)
                p_str = " ".join(p_str)
                p_str = p_str.strip() 
                p_str += '\n'
                writer.write(p_str)


        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, 'w', encoding='utf8') as writer:
            logger.info("***** Eval results *****")
            logger.info("Bleu : %s", str(bleu))
            logger.info("accuracy : %s", str(result_report['accuracy']))
            logger.info("50p_accuracy : %s", str(result_report['50p_accuracy']))
            logger.info("exact_match_accuracy : %s", str(result_report['exact_match_accuracy']))
            logger.info("exception : %s", str(result_report['other_exception_freq']))

            writer.write("Bleu : %s \n" % str(bleu))
            writer.write("accuracy : %s \n" % str(result_report['accuracy']))
            writer.write("50p_accuracy : %s \n" % str(result_report['50p_accuracy']))
            writer.write("exact_match_accuracy : %s \n" % str(result_report['exact_match_accuracy']))
            writer.write("exception : %s \n" % str(result_report['other_exception_freq']))


if __name__=="__main__":
    main()




