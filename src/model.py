"""Sequence to Sequence models."""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict



class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn



class TPRAttention(nn.Module):
    """TPR soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.

    Instead of attend to the Tensor product,
    compute attention both for role vectors and filler vectors
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(TPRAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 3, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, contextF, contextR):
        """Propogate input through the network.

        input: batch x dim
        contextF: batch x sourceL x dim
        contextR: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  

        # Get attention

        attnF = torch.bmm(contextF, target).squeeze(2) 
        attnF = self.sm(attnF)
        attn3F = attnF.view(attnF.size(0), 1, attnF.size(1))  
        weighted_contextF = torch.bmm(attn3F, contextF).squeeze(1) 

        attnR = torch.bmm(contextR, target).squeeze(2) 
        attnR = self.sm(attnR)
        attn3R = attnF.view(attnR.size(0), 1, attnR.size(1)) 
        weighted_contextR = torch.bmm(attn3R, contextR).squeeze(1)  


        h_tilde = torch.cat((weighted_contextF, weighted_contextR, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attnF, attnR



class LSTMAttentionTPR(nn.Module):
    r"""A long short-term memory (LSTM) cell with tpr attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LSTMAttentionTPR, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = TPRAttention(hidden_size)

    def forward(self, input, hidden, ctxF, ctxR, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy) 
            h_tilde, alphaF, alphaR = self.attention_layer(hy, ctxF.transpose(0, 1), ctxR.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(hidden[0])

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class LSTMAttentionDot(nn.Module):
    #A long short-term memory (LSTM) cell with attention.

    def __init__(self, input_size, hidden_size, batch_first=True, dropout=0.):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            input = self.dropout(input)
            hx, cx = hidden  
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(hidden[0])

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden




class TPN2F(nn.Module):
    """
    TPN2F model:
    src_emb_dim: natural language source embedding dimension
    trg_emb_dim: target relational tuple embedding dimension (use same dimension for relation and arguments)
    src_vocab_size: source vocabulary size
    opt_vocab_size: relation (operator) vocabulary size
    arg_vocab_size: argument (parameter) vocabulary size
    pad_token_src: source pad token id
    pad_token_opt: relation (operator) pad token id
    pad_token_arg: argument (parameter) pad token id
    attention_mode: dot/tpr. We did not get significant difference between each other on this task.
    bidirectonal: True/False
    nlayers: number of encoder layers
    nlayers_trg:number of decoder layers (multiple layers not implemented yet)
    nSymbols: number of fillers (symbols) for encoder to compute softmax scores
    nRoles: number of roles for encoder to compute softmax scores
    dSymbols: dimension of filler vector in encoder
    dRoles: dimension of role vector in encoder
    temperature: with smaller temperature, the influence on specific role or filler increases.
    dOpts: dimension of relaiton (operator) vector in decoder
    dArgs: dimension of argument (parameter) vector in decoder
    dPoss: dimension of position vector in decoder
    role_grad: True/False. Whether the position vector will be trained or one-hot vectors
    seq_len: sequnece length
    sum_T: True - sum all T in encoder as encoder output. False - only the last T is encoder output.
           Based on experiments, sum all T gets better results.
    reason_T: one/two layers of reasoning MLP
    binary: binary relation tuple or relation tuple with three arguments
    """

    def __init__(
        self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        opt_vocab_size,
        arg_vocab_size,
        pad_token_src,
        pad_token_opt,
        pad_token_arg,
        attention_mode = 'dot',
        bidirectional=False,
        nlayers=1,
        nlayers_trg=1,
        dropout=0.,
        nSymbols=150, nRoles=50, dSymbols=10, dRoles=10,
        temperature=1.0, dOpts=10, dArgs=10, dPoss=3, role_grad=True,
        seq_len = 128,
        sum_T = True, reason_T = 1,
        binary = False, temp_increase=False
    ):
        """Initialize model."""
        super(TPN2F, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.opt_vocab_size = opt_vocab_size
        self.arg_vocab_size = arg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = dSymbols*dRoles
        self.attention_mode = attention_mode
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_opt = pad_token_opt
        self.pad_token_arg = pad_token_arg
        self.sum_T = sum_T
        self.reason_T = reason_T
        self.binary = binary

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_emb_dim,
            self.pad_token_src
        )

        self.opt_embedding = nn.Embedding(
            opt_vocab_size,
            trg_emb_dim,
            self.pad_token_opt
        )

        self.arg_embedding = nn.Embedding(
            arg_vocab_size,
            trg_emb_dim,
            self.pad_token_arg
        )

        self.encoder = TPRencoder(
            self.src_emb_dim, nSymbols, nRoles, dSymbols, dRoles, temperature, nlayers, self.dropout, rnn_type='LSTM',bidirect=self.bidirectional, temp_increase=temp_increase
        )

        if binary:
            trg_emb_dim_decode = 3*trg_emb_dim
        else:
            trg_emb_dim_decode = 4*trg_emb_dim
        self.decoder = TPRdecoderForProgram(
            trg_emb_dim_decode,  attention=self.attention_mode, dOpts=dOpts, dArgs=dArgs, dPoss=dPoss, role_grad=role_grad, binary = binary,dropout=dropout
        )

        if self.reason_T > 1:
            self.encoder2decoder1 = nn.Linear(self.src_hidden_dim,dArgs*dOpts*dPoss)
            self.encoder2decoder2 = nn.Linear(dArgs*dOpts*dPoss,dArgs*dOpts*dPoss)
            # Only implement one or two layers
            # No significant improvement with more layers
        else:
            self.encoder2decoder = nn.Linear(
                self.src_hidden_dim,
                dArgs*dOpts*dPoss
            )

        if attention_mode == "dot":
            self.encoder2decoderCtx = nn.Linear(
                self.src_hidden_dim,
                dArgs*dOpts*dPoss
            )
        else:
            self.encoder2decoderCtxR = nn.Linear(
                dRoles,
                dArgs*dOpts*dPoss
            )
            self.encoder2decoderCtxF = nn.Linear(
                dSymbols,
                dArgs*dOpts*dPoss
            )
        

        self.decoder2opt = nn.Linear(dOpts,opt_vocab_size)

        self.decoder2arg = nn.Linear(dArgs,arg_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.opt_embedding.weight.data.uniform_(-initrange, initrange)
        self.arg_embedding.weight.data.uniform_(-initrange, initrange)
        if self.reason_T > 1:
            self.encoder2decoder1.bias.data.fill_(0)
        else:
            self.encoder2decoder.bias.data.fill_(0)
        self.decoder2opt.bias.data.fill_(0)
        self.decoder2arg.bias.data.fill_(0)


    def forward(self, input_src, input_trg, trg_mask=None, ctx_mask=None, rela_analysis=False):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        input_opt = input_trg[:,:,0]
        input_arg1 = input_trg[:,:,1]
        input_arg2 = input_trg[:,:,2]
        opt_emb = self.opt_embedding(input_opt)
        arg_emb1 = self.arg_embedding(input_arg1)
        arg_emb2 = self.arg_embedding(input_arg2)
        if self.binary:
            trg_emb = torch.cat((opt_emb, arg_emb1, arg_emb2),dim=2)
        else:
            input_arg3 = input_trg[:,:,3] 
            arg_emb3 = self.arg_embedding(input_arg3)
            trg_emb = torch.cat((opt_emb, arg_emb1, arg_emb2, arg_emb3),dim=2)

        out_T, aFs, aRs, (itemFs, itemRs) = self.encoder(src_emb)

        if self.attention_mode == "dot":
            ctx = self.encoder2decoderCtx(out_T).transpose(0, 1)
        else:
            ctxR = self.encoder2decoderCtxR(itemRs).transpose(0, 1)
            ctxF = self.encoder2decoderCtxF(itemFs).transpose(0, 1)
            ctx = torch.cat((ctxF.unsqueeze(-1), ctxR.unsqueeze(-1)), dim=3)


        if self.sum_T:
            out_T = torch.sum(out_T,dim=1)
        else:
            out_T = out_T[:,-1,:]

        if self.sum_T:
            out_itemR = torch.sum(itemRs,dim=1)
        else:
            out_itemR = itemRs[:,-1,:]

        if self.reason_T > 1:
            out_T = self.encoder2decoder1(out_T)
            for i in range(self.reason_T):
                out_T = nn.Tanh()(self.encoder2decoder2(out_T))
        else:
            out_T = self.encoder2decoder(out_T)
        decoder_init_state = nn.Tanh()(out_T)

        if self.binary:
            output_t, output_o, output_a1, output_a2= self.decoder(
                trg_emb,
                (decoder_init_state, out_T),
                ctx,
                ctx_mask
            )
        else:
            output_t, output_o, output_a1, output_a2, output_a3= self.decoder(
                trg_emb,
                (decoder_init_state, out_T),
                ctx,
                ctx_mask
            )
        ##############################
        output_o_reshape = output_o.contiguous().view(
            output_o.size()[0] * output_o.size()[1],
            output_o.size()[2]
        )
        decoder_logit_o = self.decoder2opt(output_o_reshape)
        decoder_logit_o = decoder_logit_o.view(
            output_o.size()[0],
            output_o.size()[1],
            decoder_logit_o.size()[1]
        )
        ##################################
        output_a1_reshape = output_a1.contiguous().view(
            output_a1.size()[0] * output_a1.size()[1],
            output_a1.size()[2]
        )
        decoder_logit_a1 = self.decoder2arg(output_a1_reshape)
        decoder_logit_a1 = decoder_logit_a1.view(
            output_a1.size()[0],
            output_a1.size()[1],
            decoder_logit_a1.size()[1]
        )
        ###################################
        output_a2_reshape = output_a2.contiguous().view(
            output_a2.size()[0] * output_a2.size()[1],
            output_a2.size()[2]
        )
        decoder_logit_a2 = self.decoder2arg(output_a2_reshape)
        decoder_logit_a2 = decoder_logit_a2.view(
            output_a2.size()[0],
            output_a2.size()[1],
            decoder_logit_a2.size()[1]
        )
        ###################################
        if self.binary:
            if rela_analysis:
                return decoder_logit_o, decoder_logit_a1, decoder_logit_a2, output_o, aFs, aRs
            else:
                return decoder_logit_o, decoder_logit_a1, decoder_logit_a2, aFs, aRs
        else:
            output_a3_reshape = output_a3.contiguous().view(
                output_a3.size()[0] * output_a3.size()[1],
                output_a3.size()[2]
            )
            decoder_logit_a3 = self.decoder2arg(output_a3_reshape)
            decoder_logit_a3 = decoder_logit_a3.view(
                output_a3.size()[0],
                output_a3.size()[1],
                decoder_logit_a3.size()[1]
            )
            if rela_analysis:
                return decoder_logit_o, decoder_logit_a1, decoder_logit_a2, decoder_logit_a3, output_o, aFs, aRs
            else:
                return decoder_logit_o, decoder_logit_a1, decoder_logit_a2, decoder_logit_a3, aFs, aRs



    def decode(self, logits, trg_type='opt'):
        """Return probability distribution over words."""
        if trg_type=='opt':
            logits_reshape = logits.view(-1, self.opt_vocab_size)
        else:
            logits_reshape = logits.view(-1, self.arg_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs



class TPRencoder(nn.Module):
    def __init__(self, in_dim, nSymbols, nRoles, dSymbols, dRoles, temperature, nlayers, dropout, bidirect = False, rnn_type='LSTM',temp_increase=False):
        """
        TP-N2F encoder for encoding natural language sequence.
        """
        super(TPRencoder, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        if rnn_type == 'LSTM':
            rnn_cls = nn.LSTM
        else:
            rnn_cls = nn.GRU

        self.nSymbols = nSymbols
        self.nRoles = nRoles
        self.dSymbols = dSymbols
        self.dRoles = dRoles
        self.temperature = temperature
        self.temp_increase = temp_increase
        self.dropout = nn.Dropout(dropout)
        if bidirect == True:
            self.out_dim = dSymbols*dRoles*2
        else:
            self.out_dim = dSymbols*dRoles
        self.rnn_aF = rnn_cls(
            in_dim, dSymbols*dRoles, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)
        self.rnn_aR = rnn_cls(
            in_dim, dSymbols*dRoles, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
        self.F = nn.Linear(nSymbols, dSymbols)
        self.R = nn.Linear(nRoles, dRoles)
        self.WaF = nn.Linear(self.out_dim, nSymbols)
        self.WaR = nn.Linear(self.out_dim, nRoles)
        self.softmax = nn.Softmax(dim=2)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers*self.ndirections, batch, self.dSymbols*self.dRoles)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]

        batch = x.size(0)
        seq = x.size(1)
        hidden_aF = self.init_hidden(batch)
        hidden_aR = self.init_hidden(batch)
        self.rnn_aF.flatten_parameters()
        self.rnn_aR.flatten_parameters()
        for i in range(seq):
            aF, hidden_aF = self.rnn_aF(x[:, i, :].unsqueeze(1), hidden_aF)
            aR, hidden_aR = self.rnn_aR(x[:, i, :].unsqueeze(1), hidden_aR)
            aF = self.WaF(aF)
            aR = self.WaR(aR)
            aF = self.softmax(aF/self.temperature)
            aR = self.softmax(aR/self.temperature)
            itemF = self.F(aF)
            itemR = self.R(aR)
            itemF = itemF.view([batch,self.dSymbols,1])
            T = (torch.bmm(itemF, itemR)).view(batch, -1)
            T = self.dropout(T)
            if self.temp_increase:
                T = T/self.temperature
            T_next = T.unsqueeze(0)
            for j in range(self.nlayers*self.ndirections-1):
                T_next = torch.cat((T_next,T.unsqueeze(0)))
            hidden_aF = (T_next, hidden_aF[1])
            hidden_aR = (T_next, hidden_aR[1])
            if i==0:
                out = T.unsqueeze(1)
                aFs = aF
                aRs = aR
                itemFs = itemF.view(-1, itemF.size(2), itemF.size(1))
                itemRs = itemR
            else:
                out = torch.cat([out, T.unsqueeze(1)], 1)
                aFs = torch.cat([aFs, aF], 1)
                aRs = torch.cat([aRs, aR], 1)
                itemF = itemF.view(-1, itemF.size(2), itemF.size(1))
                itemFs = torch.cat([itemFs, itemF], 1)
                itemRs = torch.cat([itemRs, itemR], 1)

        return out, aFs, aRs, (itemFs, itemRs)


class TPRdecoderForProgram(nn.Module):
    def __init__(self, in_dim, attention="dot", dOpts=30, dArgs=20, dPoss = 5, nlayers=1, dropout=0., bidirect = False, role_grad=True, binary=False):
        """
        TP-N2F decoder for generate program tuple.
        """
        super(TPRdecoderForProgram, self).__init__()
        self.dOpts = dOpts
        self.dArgs = dArgs
        self.dPoss = dPoss
        self.T_out_dim = dArgs*dOpts*dPoss
        self.in_dim = in_dim
        self.nlayers = nlayers
        self.ndirections = 1 + int(bidirect)
        self.attention = attention
        self.binary = binary
        if attention == "dot":
            self.rnn_T = LSTMAttentionDot(
                in_dim,
                self.T_out_dim,
                batch_first=True,
                dropout=dropout
            )
        else:
            self.rnn_T = LSTMAttentionTPR(
                in_dim,
                self.T_out_dim,
                batch_first=True
            )
        self.O_plus = nn.Linear(dArgs*dOpts, dOpts)
        #self.O = nn.Linear(dOpts, dOpts)
        if role_grad == False:
            self.ur1 = Variable(torch.Tensor([1,0,0]), requires_grad=role_grad)
            self.ur2 = Variable(torch.Tensor([0,1,0]), requires_grad=role_grad)
            if not binary:
                self.ur3 = Variable(torch.Tensor([0,0,1]), requires_grad=role_grad)
        else:
            self.ur1 = nn.Parameter(torch.Tensor(dPoss), requires_grad=role_grad)
            self.ur2 = nn.Parameter(torch.Tensor(dPoss), requires_grad=role_grad)
            self.ur1.data.uniform_(0.0,1.0)
            self.ur2.data.uniform_(0.0,1.0)
            if not binary:
                self.ur3 = nn.Parameter(torch.Tensor(dPoss), requires_grad=role_grad)
                self.ur3.data.uniform_(0.0,1.0)

    def init_hidden(self, batch):
        # just to get the type of tensor
        tt_shape = (self.nlayers*self.ndirections, batch, self.T_out_dim)
        return (torch.zeros(tt_shape,requires_grad=True),torch.zeros(tt_shape,requires_grad=True))


    def forward(self, x, hidden, ctx, ctx_mask=None):
        # x: [batch, sequence, emb_dim]
        # hidden[0]: [batch, dxdxd]
        # hidden[1]: [batch, dxdxd]
        batch = x.size(0)
        seq = x.size(1)
        if hidden == None:
            hidden_tT= self.init_hidden(batch)
        else:
            hidden_tT= hidden

        if self.attention == 'dot':
            tT, hidden_tT = self.rnn_T(x, hidden_tT, ctx)
        else:
            ctxF = ctx[:, :, :, 0]
            ctxR = ctx[:, :, :, 1]
            tT, hidden_tT = self.rnn_T(x, hidden_tT, ctxF, ctxR)

        tT_next_reshape = tT.contiguous().view(batch, seq, self.dArgs, self.dOpts, self.dPoss)

        a1o = torch.matmul(tT_next_reshape,self.ur1)
        a2o = torch.matmul(tT_next_reshape,self.ur2)
        if not self.binary:
            a3o = torch.matmul(tT_next_reshape,self.ur3)
            ao_next = a1o+a2o+a3o
        else:
            ao_next = a1o+a2o
        
        ao_next_reshape = ao_next.contiguous().view(batch,seq,-1)
        to_plus = self.O_plus(ao_next_reshape)
        word_probs = F.softmax(to_plus)
        #to_plus = nn.Tanh()(self.O_plus(ao_next_reshape))
        #to = self.O(to_plus)

        a1 = torch.matmul(a1o,to_plus.unsqueeze(-1)).squeeze(-1)
        a2 = torch.matmul(a2o,to_plus.unsqueeze(-1)).squeeze(-1)
        if not self.binary:
            a3 = torch.matmul(a3o,to_plus.unsqueeze(-1)).squeeze(-1)
            return (tT, to_plus, a1, a2, a3)
        else:
            return (tT, to_plus, a1, a2)