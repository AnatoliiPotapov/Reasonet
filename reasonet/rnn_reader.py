# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers
from .biattnflow import BiAttentionLayer, masked_softmax
# Modification: add 'pos' and 'ner' features.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding(embedding.size(0),
                                          embedding.size(1),
                                          padding_idx=padding_idx)
            self.embedding.weight.data = embedding
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
        if opt['pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
        if opt['ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['features']*opt['num_features']

        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_dim']
        if opt['ner']:
            doc_input_size += opt['ner_dim']

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=opt['embedding_dim'],
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Bidirectional Attention Flow
        self.biattnflow = BiAttentionLayer()

        # Context modelling
        self.M1_encoder = layers.StackedBRNN(
            input_size=3072,
            hidden_size=150,
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=False,
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # Context modelling
        self.M2_encoder = layers.StackedBRNN(
            input_size=300,
            hidden_size=150,
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=False,
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # Answer pointers
        self.start_pointer = layers.AnswerPointer()
        self.end_pointer = layers.AnswerPointer()


    def forward(self, x1, x1_mask, x2, x2_mask, x1_f = None, x1_pos = None, x1_ner = None):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                           training=self.training)

        drnn_input_list = [x1_emb]

        if self.opt['features']:
            drnn_input_list.append(x1_f)
        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            drnn_input_list.append(x1_ner_emb)
        drnn_input = torch.cat(drnn_input_list, 2)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)

        # Make Attention Flow
        biattn = self.biattnflow(doc_hiddens, question_hiddens, x1_mask, x2_mask)
        print(biattn.size())

        # Context modelling layer 1
        M1 = self.M1_encoder(biattn, x1_mask)

        # Context modelling layer 2
        M2 = self.M2_encoder(M1, x1_mask)

        start_scores = masked_softmax(self.start_pointer(M1), x1_mask.type(torch.FloatTensor))
        end_scores = masked_softmax(self.end_pointer(M2), x1_mask.type(torch.FloatTensor))

        return start_scores, end_scores
