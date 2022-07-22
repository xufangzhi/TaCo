import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from tools import allennlp as util
from transformers import BertPreTrainedModel, RobertaModel, BertModel, RobertaTokenizer,RobertaForMultipleChoice
import torch.nn.functional as F
import math

class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.attn_bias_linear = nn.Linear(1, self.num_heads)

    def forward(self, q, k, v, attn_bias=None, attention_mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(-1).permute(0, 3, 1, 2)
            attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)
            x += attn_bias

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(0, 3, 1, 2)
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            x += attention_mask

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias=None):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, attention_mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, attention_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

# normal position embedding
class Position_Embedding(nn.Module):
    def __init__(self, hidden_size):
        super(Position_Embedding, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, x):  # input is encoded spans
        batch_size = x.size(0)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)  # [seq_len] -> [batch_size, seq_len]
        self.pos_embed = nn.Embedding(seq_len, self.hidden_size)  # position embedding
        embedding = self.pos_embed(pos)

        return embedding.to(x.device)


class ZsLR(BertPreTrainedModel):
    def __init__(self, config, des_embedding,):
        super().__init__(config)
        self.logical_info = False
        self.two_subgraphs = False
        self.one_graph = True
        self.loss = "margin"
        self.des_embedding=des_embedding.float()
        self.layer_num = 5
        self.head_num = 5
        self.dropout_prob = 0.1
        self.hidden_size = 1024
        self.gamma = 12
        self.max_rel_id = 4 
        self.classifier = nn.Linear(self.hidden_size, 1)

        if self.logical_info:
            self.pos_embed = Position_Embedding(self.hidden_size)
            self.input_dropout = nn.Dropout(self.dropout_prob)
            encoders = [EncoderLayer(self.hidden_size, self.hidden_size, self.dropout_prob, self.dropout_prob, self.head_num)
                        for _ in range(self.layer_num)]
            self.encoder_layers = nn.ModuleList(encoders)
            self.dropout = nn.Dropout(self.dropout_prob)
            self.final_ln = nn.LayerNorm(self.hidden_size)
            self._prj_ln = nn.LayerNorm(self.hidden_size)
            self._enc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU())
            self._proj_sequence_h = nn.Linear(self.hidden_size, 1, bias=False)
            self.classifier_merge = nn.Linear(4*self.hidden_size, 1)

        if self.two_subgraphs:
            self.pos_embed = Position_Embedding(self.hidden_size)
            self.input_dropout = nn.Dropout(self.dropout_prob)
            encoders = [EncoderLayer(self.hidden_size, self.hidden_size, self.dropout_prob, self.dropout_prob, self.head_num)
                        for _ in range(self.layer_num)]
            self.encoder_layers = nn.ModuleList(encoders)
            self.dropout = nn.Dropout(self.dropout_prob)
            self.final_ln = nn.LayerNorm(self.hidden_size)
            self._prj_ln = nn.LayerNorm(self.hidden_size)
            self._enc_context = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU())
            self._enc_qa = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU())
            self._proj_sequence_h = nn.Linear(self.hidden_size, 1, bias=False)
            self.classifier_merge = nn.Linear(4*self.hidden_size, 1)
            self.feedforward =  nn.Sequential(nn.Linear(2*self.hidden_size, self.hidden_size, bias=False), nn.ReLU())
        
        if self.one_graph:
            self.pos_embed = Position_Embedding(self.hidden_size)
            self.input_dropout = nn.Dropout(self.dropout_prob)
            encoders = [EncoderLayer(self.hidden_size, self.hidden_size, self.dropout_prob, self.dropout_prob, self.head_num)
                        for _ in range(self.layer_num)]
            self.encoder_layers = nn.ModuleList(encoders)
            self.dropout = nn.Dropout(self.dropout_prob)
            self.final_ln = nn.LayerNorm(self.hidden_size)
            self._prj_ln = nn.LayerNorm(self.hidden_size)
            self._enc_context = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU())
            self._enc_qa = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU())
            self._proj_sequence_h = nn.Linear(self.hidden_size, 1, bias=False)
            self.classifier_merge = nn.Linear(4*self.hidden_size, 1)


        ''' from modeling_roberta '''
        self.roberta = RobertaModel(config)
    

    def split_into_subgraphs(self, seq, seq_mask, split_bpe_ids, passage_mask, option_mask, type):
        '''
        for context graph and QA graph, obtain the encoded spans respectively.
        '''
        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        context_encoded_spans, qa_encoded_spans = [], []
        context_span_masks, qa_span_masks = [], []
        context_edges, qa_edges = [], []
        context_edges_embed, qa_edges_embed = [], []
        context_node_in_seq_indices, qa_node_in_seq_indices = [], []
        context_embed, qa_embed = [], []

        for item_seq_mask, item_seq, item_split_ids, p_mask, o_mask in zip(seq_mask, seq, split_bpe_ids, passage_mask, option_mask):
            item_seq_len = item_seq_mask.sum().item()    # sequence length
            item_context_len = p_mask.sum().item()   # context length
            item_context = item_seq[:item_context_len]     # context
            item_qa = item_seq[item_context_len:item_seq_len]     # qa
            item_context_split_ids = item_split_ids[:item_context_len]
            item_qa_split_ids = item_split_ids[item_context_len:item_seq_len]
            item_context_split_ids = item_context_split_ids.cpu().numpy()
            item_qa_split_ids = item_qa_split_ids.cpu().numpy()
            context_split_ids_indices = np.where(item_context_split_ids > 0)[0].tolist()     # consider both punctuations and connective words
            qa_split_ids_indices = np.where(item_qa_split_ids > 0)[0].tolist()     # consider both punctuations and connective words

            context_grouped_split_ids_indices, context_split_ids_indices, item_context_split_ids = _consecutive(
                context_split_ids_indices, item_context_split_ids)
            qa_grouped_split_ids_indices, qa_split_ids_indices, item_qa_split_ids = _consecutive(
                qa_split_ids_indices, item_qa_split_ids)

            context_n_split_ids = len(context_split_ids_indices)   # the number of split ids in context graph 
            qa_n_split_ids = len(qa_split_ids_indices)     # the number of split ids in qa graph 

            item_context_spans, item_qa_spans = [], []
            item_context_mask, item_qa_mask = [], []
            item_context_edges, item_qa_edges = [], []
            item_context_node_in_seq_indices, item_qa_node_in_seq_indices = [], []
            item_context_edges.append(item_context_split_ids[context_split_ids_indices[0]])
            item_qa_edges.append(item_qa_split_ids[qa_split_ids_indices[0]])

            for i in range(context_n_split_ids):      # get context graph node
                if i == context_n_split_ids - 1:
                    span = item_context[context_split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_context_spans.append(span.sum(0))
                        item_context_mask.append(1)
                else:
                    span = item_context[context_split_ids_indices[i] + 1:context_split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_context_spans.append(span.sum(0))  # span.sum(0) calculate the sum of embedding value at each position (1024 in total)
                        item_context_mask.append(1)
                        item_context_edges.append(item_context_split_ids[context_split_ids_indices[i + 1]])  # the edge type after the span
                        item_context_node_in_seq_indices.append([i for i in range(context_grouped_split_ids_indices[i][-1] + 1, context_grouped_split_ids_indices[i + 1][0])])  
                                                                                        # node indices [[1, 2], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]....]
            for i in range(qa_n_split_ids):      # get qa graph node
                if i == qa_n_split_ids - 1:
                    span = item_qa[qa_split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_qa_spans.append(span.sum(0))
                        item_qa_mask.append(1)
                    if len(qa_split_ids_indices) == 1:
                        item_qa_spans.append(item_qa.sum(0))
                        item_qa_mask.append(1)

                else:
                    span = item_qa[qa_split_ids_indices[i] + 1:qa_split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_qa_spans.append(span.sum(0))  # span.sum(0) calculate the sum of embedding value at each position (1024 in total)
                        item_qa_mask.append(1)
                        item_qa_edges.append(item_qa_split_ids[qa_split_ids_indices[i + 1]])  # the edge type after the span
                        item_qa_node_in_seq_indices.append([i for i in range(qa_grouped_split_ids_indices[i][-1] + 1, qa_grouped_split_ids_indices[i + 1][0])])  
                                                                                        # node indices [[1, 2], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]....]
            context_encoded_spans.append(item_context_spans)
            context_span_masks.append(item_context_mask)
            context_edges.append(item_context_edges)
            context_node_in_seq_indices.append(item_context_node_in_seq_indices)
            context_embed.append(item_context)
            qa_encoded_spans.append(item_qa_spans)
            qa_span_masks.append(item_qa_mask)
            qa_edges.append(item_qa_edges)
            qa_node_in_seq_indices.append(item_qa_node_in_seq_indices)
            qa_embed.append(item_qa)

        ''' output for context graph'''
        context_max_nodes = max(map(len, context_span_masks))  # span_masks:[n_choice * batch_size, node_num]
        context_span_masks = [spans + [0] * (context_max_nodes - len(spans)) for spans in
                      context_span_masks]  # make the node number be the same
        context_span_masks = torch.from_numpy(np.array(context_span_masks))
        context_span_masks = context_span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        context_attention_mask = torch.zeros((seq.size(0), context_max_nodes+1, context_max_nodes+1), dtype=seq.dtype, device=seq.device) # max_nodes+1 is an additional global node
        context_attention_mask += -1e9
        for i, spans in enumerate(context_encoded_spans):
            context_attention_mask[i, :, :len(spans)+1] = 0
        context_encoded_spans = [spans + [pad_embed] * (context_max_nodes - len(spans)) for spans in context_encoded_spans]  # [n_choice * batch_size, max_node_num, hidden_size]
        context_encoded_spans = [torch.stack(lst, dim=0) for lst in context_encoded_spans]
        context_encoded_spans = torch.stack(context_encoded_spans, dim=0)
        context_encoded_spans = context_encoded_spans.to(device).float()  # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        # Truncate head and tail of each list in edges HERE.
        # Because the head and tail edge DO NOT contribute to the argument graph and punctuation graph.
        context_truncated_edges = [item[1:-1] for item in context_edges]

        max_context_len = max([p_mask.sum().item() for p_mask in passage_mask])
        context_embed_list = []
        for spans in context_embed:
            if max_context_len != spans.size(0):
                context_pad_embed = pad_embed.unsqueeze(0).repeat(max_context_len-spans.size(0), 1)
                inputs = torch.cat([spans, context_pad_embed], dim=0)
            else:
                inputs = spans
            context_embed_list.append(inputs)
        context_embed = torch.stack(context_embed_list, dim=0).to(device).float()

        ''' output for qa graph'''
        qa_max_nodes = max(map(len, qa_span_masks))  # span_masks:[n_choice * batch_size, node_num]
        qa_span_masks = [spans + [0] * (qa_max_nodes - len(spans)) for spans in
                      qa_span_masks]  # make the node number be the same
        qa_span_masks = torch.from_numpy(np.array(qa_span_masks))
        qa_span_masks = qa_span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        qa_attention_mask = torch.zeros((seq.size(0), qa_max_nodes+1, qa_max_nodes+1), dtype=seq.dtype, device=seq.device) # max_nodes+1 is an additional global node
        qa_attention_mask += -1e9
        for i, spans in enumerate(qa_encoded_spans):
            qa_attention_mask[i, :, :len(spans)+1] = 0
        qa_encoded_spans = [spans + [pad_embed] * (qa_max_nodes - len(spans)) for spans in qa_encoded_spans]  # [n_choice * batch_size, max_node_num, hidden_size]
        qa_encoded_spans = [torch.stack(lst, dim=0) for lst in qa_encoded_spans]
        qa_encoded_spans = torch.stack(qa_encoded_spans, dim=0)
        qa_encoded_spans = qa_encoded_spans.to(device).float()  # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        qa_truncated_edges = [item[1:-1] for item in qa_edges]

        max_qa_len = max([o_mask.sum().item() for o_mask in option_mask])
        qa_embed_list = []
        for spans in qa_embed:
            if max_qa_len != spans.size(0):
                qa_pad_embed = pad_embed.unsqueeze(0).repeat(max_qa_len-spans.size(0), 1)
                inputs = torch.cat([spans, qa_pad_embed], dim=0)
            else:
                inputs = spans
            qa_embed_list.append(inputs)
        qa_embed = torch.stack(qa_embed_list, dim=0).to(device).float()
        return context_encoded_spans, context_truncated_edges, context_node_in_seq_indices, context_attention_mask, context_embed, \
               qa_encoded_spans, qa_truncated_edges, qa_node_in_seq_indices, qa_attention_mask, qa_embed

    def split_into_one_graph(self, seq, seq_mask, split_bpe_ids, passage_mask, option_mask, type):
        '''
        for context graph and QA graph, obtain the encoded spans respectively.
        '''
        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        context_encoded_spans, qa_encoded_spans = [], []
        context_span_masks, qa_span_masks = [], []
        context_edges, qa_edges = [], []
        context_edges_embed, qa_edges_embed = [], []
        context_node_in_seq_indices, qa_node_in_seq_indices = [], []
        context_embed, qa_embed = [], []

        for item_seq_mask, item_seq, item_split_ids, p_mask, o_mask in zip(seq_mask, seq, split_bpe_ids, passage_mask, option_mask):
            item_seq_len = item_seq_mask.sum().item()    # sequence length
            item_context_len = p_mask.sum().item()   # context length
            item_context = item_seq[:item_context_len]     # context
            item_qa = item_seq[item_context_len:item_seq_len]     # qa
            item_context_split_ids = item_split_ids[:item_context_len]
            item_qa_split_ids = item_split_ids[item_context_len:item_seq_len]
            item_context_split_ids = item_context_split_ids.cpu().numpy()
            item_qa_split_ids = item_qa_split_ids.cpu().numpy()
            context_split_ids_indices = np.where(item_context_split_ids > 0)[0].tolist()     # consider both punctuations and connective words
            qa_split_ids_indices = np.where(item_qa_split_ids > 0)[0].tolist()     # consider both punctuations and connective words

            context_grouped_split_ids_indices, context_split_ids_indices, item_context_split_ids = _consecutive(
                context_split_ids_indices, item_context_split_ids)
            qa_grouped_split_ids_indices, qa_split_ids_indices, item_qa_split_ids = _consecutive(
                qa_split_ids_indices, item_qa_split_ids)

            context_n_split_ids = len(context_split_ids_indices)   # the number of split ids in context graph 
            qa_n_split_ids = len(qa_split_ids_indices)     # the number of split ids in qa graph 

            item_context_spans, item_qa_spans = [], []
            item_context_mask, item_qa_mask = [], []
            item_context_edges, item_qa_edges = [], []
            item_context_node_in_seq_indices, item_qa_node_in_seq_indices = [], []
            item_context_edges.append(item_context_split_ids[context_split_ids_indices[0]])
            item_qa_edges.append(item_qa_split_ids[qa_split_ids_indices[0]])

            for i in range(context_n_split_ids):      # get context graph node
                if i == context_n_split_ids - 1:
                    span = item_context[context_split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_context_spans.append(span.sum(0))
                        item_context_mask.append(1)
                else:
                    span = item_context[context_split_ids_indices[i] + 1:context_split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_context_spans.append(span.sum(0))  # span.sum(0) calculate the sum of embedding value at each position (1024 in total)
                        item_context_mask.append(1)
                        item_context_edges.append(item_context_split_ids[context_split_ids_indices[i + 1]])  # the edge type after the span
                        item_context_node_in_seq_indices.append([i for i in range(context_grouped_split_ids_indices[i][-1] + 1, context_grouped_split_ids_indices[i + 1][0])])  
                                                                                        # node indices [[1, 2], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]....]
            for i in range(qa_n_split_ids):      # get qa graph node
                if i == qa_n_split_ids - 1:
                    span = item_qa[qa_split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_qa_spans.append(span.sum(0))
                        item_qa_mask.append(1)
                    if len(qa_split_ids_indices) == 1:
                        item_qa_spans.append(item_qa.sum(0))
                        item_qa_mask.append(1)

                else:
                    span = item_qa[qa_split_ids_indices[i] + 1:qa_split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_qa_spans.append(span.sum(0))  # span.sum(0) calculate the sum of embedding value at each position (1024 in total)
                        item_qa_mask.append(1)
                        item_qa_edges.append(item_qa_split_ids[qa_split_ids_indices[i + 1]])  # the edge type after the span
                        item_qa_node_in_seq_indices.append([i for i in range(qa_grouped_split_ids_indices[i][-1] + 1, qa_grouped_split_ids_indices[i + 1][0])])  
                                                                                        # node indices [[1, 2], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]....]
            context_encoded_spans.append(item_context_spans)
            context_span_masks.append(item_context_mask)
            context_edges.append(item_context_edges)
            context_node_in_seq_indices.append(item_context_node_in_seq_indices)
            context_embed.append(item_context)
            qa_encoded_spans.append(item_qa_spans)
            qa_span_masks.append(item_qa_mask)
            qa_edges.append(item_qa_edges)
            qa_node_in_seq_indices.append(item_qa_node_in_seq_indices)
            qa_embed.append(item_qa)


        ''' output for context graph'''
        context_max_nodes = max(map(len, context_span_masks))  # span_masks:[n_choice * batch_size, node_num]
        context_span_masks = [spans + [0] * (context_max_nodes - len(spans)) for spans in
                      context_span_masks]  # make the node number be the same
        context_span_masks = torch.from_numpy(np.array(context_span_masks))
        context_span_masks = context_span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)

        context_encoded_spans = [spans + [pad_embed] * (context_max_nodes - len(spans)) for spans in context_encoded_spans]  # [n_choice * batch_size, max_node_num, hidden_size]
        context_encoded_spans = [torch.stack(lst, dim=0) for lst in context_encoded_spans]
        context_encoded_spans = torch.stack(context_encoded_spans, dim=0)
        context_encoded_spans = context_encoded_spans.to(device).float()  # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        # Truncate head and tail of each list in edges HERE.
        # Because the head and tail edge DO NOT contribute to the argument graph and punctuation graph.
        context_truncated_edges = [item[1:-1] for item in context_edges]

        max_node_len = max([node_mask.sum().item() for node_mask in passage_mask+option_mask])
        context_embed_list = []
        for spans in context_embed:
            if max_node_len != spans.size(0):
                context_pad_embed = pad_embed.unsqueeze(0).repeat(max_node_len-spans.size(0), 1)
                inputs = torch.cat([spans, context_pad_embed], dim=0)
            else:
                inputs = spans
            context_embed_list.append(inputs)
        context_embed = torch.stack(context_embed_list, dim=0).to(device).float()

        ''' output for qa graph'''
        qa_max_nodes = max(map(len, qa_span_masks))  # span_masks:[n_choice * batch_size, node_num]
        qa_span_masks = [spans + [0] * (qa_max_nodes - len(spans)) for spans in
                      qa_span_masks]  # make the node number be the same
        qa_span_masks = torch.from_numpy(np.array(qa_span_masks))
        qa_span_masks = qa_span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)

        qa_encoded_spans = [spans + [pad_embed] * (qa_max_nodes - len(spans)) for spans in qa_encoded_spans]  # [n_choice * batch_size, max_node_num, hidden_size]
        qa_encoded_spans = [torch.stack(lst, dim=0) for lst in qa_encoded_spans]
        qa_encoded_spans = torch.stack(qa_encoded_spans, dim=0)
        qa_encoded_spans = qa_encoded_spans.to(device).float()  # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        qa_truncated_edges = [item[1:-1] for item in qa_edges]

        qa_embed_list = []
        for spans in qa_embed:
            if max_node_len != spans.size(0):
                qa_pad_embed = pad_embed.unsqueeze(0).repeat(max_node_len-spans.size(0), 1)
                inputs = torch.cat([spans, qa_pad_embed], dim=0)
            else:
                inputs = spans
            qa_embed_list.append(inputs)
        qa_embed = torch.stack(qa_embed_list, dim=0).to(device).float()

        encoded_spans = torch.cat([context_encoded_spans,qa_encoded_spans], dim=1)
        attention_mask = torch.zeros((seq.size(0), qa_max_nodes+context_max_nodes+1, qa_max_nodes+context_max_nodes+1), dtype=seq.dtype, device=seq.device) # max_nodes+1 is an additional global node
        attention_mask += -1e9
        for i, spans in enumerate(encoded_spans):
            attention_mask[i, :, :len(spans)+1] = 0

        return context_encoded_spans, context_truncated_edges, context_node_in_seq_indices, attention_mask, context_embed, \
               qa_encoded_spans, qa_truncated_edges, qa_node_in_seq_indices, attention_mask, qa_embed


    def split_into_spans_9(self, seq, seq_mask, split_bpe_ids, passage_mask, option_mask, type):
        '''
            :param seq: (bsz, seq_length, embed_size)
            :param seq_mask: (bsz, seq_length)
            :param split_bpe_ids: (bsz, seq_length). value = {-1, 0, 1, 2, 3, 4}.
            :return:
                - encoded_spans: (bsz, n_nodes, embed_size)
                - span_masks: (bsz, n_nodes)
                - edges: (bsz, n_nodes - 1)
                - node_in_seq_indices: list of list of list(len of span).
        '''

        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        encoded_spans = []
        span_masks = []
        edges = []
        edges_embed = []
        node_in_seq_indices = []
        for item_seq_mask, item_seq, item_split_ids, p_mask, o_mask in zip(seq_mask, seq, split_bpe_ids, passage_mask, option_mask,):
            item_seq_len = item_seq_mask.sum().item()
            item_seq = item_seq[:item_seq_len]
            item_split_ids = item_split_ids[:item_seq_len]
            item_split_ids = item_split_ids.cpu().numpy()
            if type == "causal":
                split_ids_indices = np.where(item_split_ids > 0)[0].tolist()     # Causal type
            else:
                split_ids_indices = np.where(item_split_ids > 0)[0].tolist()     # Co-reference type

            grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
                split_ids_indices, item_split_ids)
            # print(grouped_split_ids_indices)     [[0], [3], [14, 15, 16], [23], [28], [32], [34], [46], [58], [66, 67], [71], [81], [101, 102]]
            # print(split_ids_indices)   [0, 3, 14, 23, 28, 32, 34, 46, 58, 66, 71, 81, 101]
            # print(item_split_ids)
            # [5 0 0 5 0 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0 5 0 0 0 0 5 0 0 0 5 0 4 0 0
            #    0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 5 5 0 0 0 4 0 0
            #    0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5]

            n_split_ids = len(split_ids_indices)

            item_spans, item_mask = [], []
            item_edges = []
            item_edges_embed = []
            item_node_in_seq_indices = []
            item_edges.append(item_split_ids[split_ids_indices[0]])
            for i in range(n_split_ids):
                if i == n_split_ids - 1:
                    span = item_seq[split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)

                else:
                    span = item_seq[split_ids_indices[i] + 1:split_ids_indices[i + 1]]
                    # span = item_seq[grouped_split_ids_indices[i][-1] + 1:grouped_split_ids_indices[i + 1][0]]
                    if not len(span) == 0:
                        item_spans.append(span.sum(
                            0))  # span.sum(0) calculate the sum of embedding value at each position (1024 in total)
                        item_mask.append(1)
                        item_edges.append(item_split_ids[split_ids_indices[i + 1]])  # the edge type after the span
                        item_edges_embed.append(item_seq[split_ids_indices[i + 1]])  # the edge embedding after the span
                        item_node_in_seq_indices.append([i for i in range(grouped_split_ids_indices[i][-1] + 1,
                                                                          grouped_split_ids_indices[i + 1][
                                                                              0])])  # node indices [[1, 2], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]....]
            encoded_spans.append(item_spans)
            span_masks.append(item_mask)
            edges.append(item_edges)
            edges_embed.append(item_edges_embed)
            node_in_seq_indices.append(item_node_in_seq_indices)

        max_nodes = max(map(len, span_masks))  # span_masks:[n_choice * batch_size, node_num]
        span_masks = [spans + [0] * (max_nodes - len(spans)) for spans in
                      span_masks]  # make the node number be the same
        span_masks = torch.from_numpy(np.array(span_masks))
        span_masks = span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        attention_mask = torch.zeros((seq.size(0), max_nodes, max_nodes), dtype=seq.dtype, device=seq.device)  
        attention_mask += -1e9
        for i, spans in enumerate(encoded_spans):
            attention_mask[i, :, :len(spans)] = 0

        encoded_spans = [spans + [pad_embed] * (max_nodes - len(spans)) for spans in
                         encoded_spans]  # [n_choice * batch_size, max_node_num, hidden_size]
        encoded_spans = [torch.stack(lst, dim=0) for lst in encoded_spans]
        encoded_spans = torch.stack(encoded_spans, dim=0)
        encoded_spans = encoded_spans.to(device).float()  # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        # Truncate head and tail of each list in edges HERE.
        #     Because the head and tail edge DO NOT contribute to the argument graph and punctuation graph.
        truncated_edges = [item[1:-1] for item in edges]
        truncated_edges_embed = [item[1:-1] for item in edges_embed]

        return encoded_spans, span_masks, truncated_edges, truncated_edges_embed, node_in_seq_indices, attention_mask

    def get_gcn_info_vector(self, indices, node, size, device):
        '''
        give the node embed to each token in one node

        :param indices: list(len=bsz) of list(len=n_notes) of list(len=varied).
        :param node: (bsz, n_nodes, embed_size)
        :param size: value=(bsz, seq_len, embed_size)
        :param device:
        :return:
        '''
        batch_size = size[0]
        gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)

        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]):
                gcn_info_vec[b, ids] = emb
            gcn_info_vec[b, 0] = node[b].mean(0)   # global feature
        return gcn_info_vec

    def get_adjacency_matrices_2(self, edges: List[List[int]], coref_tags,
                                 n_nodes: int, device: torch.device, type: str):
        '''
        Convert the edge_value_list into adjacency matrices.
            * argument graph adjacency matrix. Asymmetric (directed graph).
            * punctuation graph adjacency matrix. Symmetric (undirected graph).

            : argument
                - edges:list[list[str]]. len_out=(bsz x n_choices), len_in=n_edges. value={-1, 0, 1, 2, 3, 4, 5}.

        '''
        batch_size = len(edges)
        hidden_size = 1024
        argument_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        punct_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        casual_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if edge_value == 1:  # (relation, head, tail)  关键词在句首. Note: not used in graph_version==4.0.
                    try:
                        argument_graph[b, i + 1, i + 2] = 1
                    except Exception:
                        pass
                elif edge_value == 2:  # (head, relation, tail)  关键词在句中，先因后果. Note: not used in graph_version==4.0.
                    argument_graph[b, i, i + 1] = 1
                elif edge_value == 3:  # (tail, relation, head)  关键词在句中，先果后因. Note: not used in graph_version==4.0.
                    argument_graph[b, i + 1, i] = 1
                    casual_graph[b, i, i + 1] = 1
                    # casual_graph[b, i + 1, i] = 1
                elif edge_value == 4:  # (head, relation, tail) & (tail, relation, head) ON ARGUMENT GRAPH
                    argument_graph[b, i, i + 1] = 1
                    # argument_graph[b, i + 1, i] = 1
                elif edge_value == 5:  # (head, relation, tail) & (tail, relation, head) ON PUNCTUATION GRAPH
                    try:
                        punct_graph[b, i, i + 1] = 1
                        punct_graph[b, i + 1, i] = 1
                    except Exception:
                        pass

        ''' coref tag calculate '''
        coref_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes), dtype=torch.float)  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        if type == "coref":
            for b, sample_coref in enumerate(coref_tags):
                for i, tag in enumerate(sample_coref):
                    if tag[0].item() != -1:
                        coref_graph[b, int(tag[0].item()), int(tag[1].item())] = 1
                        coref_graph[b, int(tag[1].item()), int(tag[0].item())] = 1
            coref_graph[b, -1, :] = 1   # global node to all
            coref_graph[b, :, -1] = 1   # global node to all
        return argument_graph.to(device), punct_graph.to(device), casual_graph.to(device), coref_graph.to(device)

        # return argument_graph.to(device), punct_graph.to(device), casual_graph.to(device)

    def get_adjacency_matrices_one_graph(self, context_encoded_spans, context_coref_tags, qa_coref_tags,
                                 n_nodes: int, device: torch.device, type: str):

        batch_size = context_encoded_spans.size(0)
        context_node_len = context_encoded_spans.size(1)
        hidden_size = 1024
        ''' coref tag calculate '''
        coref_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes), dtype=torch.float)  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        if type == "coref":
            for b, sample_coref in enumerate(context_coref_tags):
                for i, tag in enumerate(sample_coref):
                    if tag[0].item() != -1:
                        coref_graph[b, int(tag[0].item()), int(tag[1].item())] = 1
                        coref_graph[b, int(tag[1].item()), int(tag[0].item())] = 1
            for b, sample_coref in enumerate(qa_coref_tags):
                for i, tag in enumerate(sample_coref):
                    if tag[0].item() != -1:
                        coref_graph[b, int(tag[0].item()+context_node_len), int(tag[1].item()+context_node_len)] = 1
                        coref_graph[b, int(tag[1].item()+context_node_len), int(tag[0].item()+context_node_len)] = 1
                coref_graph[b, -1, :] = 1   # global node to all
                coref_graph[b, :, -1] = 1   # global node to all
        return coref_graph.to(device)



    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,

                passage_mask: torch.LongTensor,
                option_mask: torch.LongTensor,

                argument_bpe_ids: torch.LongTensor,
                domain_bpe_ids: torch.LongTensor,
                punct_bpe_ids: torch.LongTensor,

                labels: torch.LongTensor,
                context_occ: torch.LongTensor,
                qa_occ: torch.LongTensor,
                qtype: torch.LongTensor,
                token_type_ids: torch.LongTensor = None,
                ) -> Tuple:

        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        flat_passage_mask = passage_mask.view(-1, passage_mask.size(
            -1)) if passage_mask is not None else None  # [num_choice*batchsize, hidden_size]
        flat_option_mask = option_mask.view(-1, option_mask.size(
            -1)) if option_mask is not None else None  # [num_choice*batchsize, hidden_size]

        flat_argument_bpe_ids = argument_bpe_ids.view(-1, argument_bpe_ids.size(
            -1)) if argument_bpe_ids is not None else None
        flat_domain_bpe_ids = domain_bpe_ids.view(-1, domain_bpe_ids.size(-1)) if domain_bpe_ids is not None else None
        flat_punct_bpe_ids = punct_bpe_ids.view(-1, punct_bpe_ids.size(-1)) if punct_bpe_ids is not None else None
        flat_context_occ = context_occ.view(-1, context_occ.size(-2), context_occ.size(-1)) if context_occ is not None else None
        flat_qa_occ = qa_occ.view(-1, qa_occ.size(-2), qa_occ.size(-1)) if qa_occ is not None else None

        flat_qtype = qtype.view(-1) if qtype is not None else None
        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None, output_hidden_states=True)

        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]
        hidden_states = bert_outputs[2]
        concat_sequence_output = hidden_states[-1] + hidden_states[-2]
        concat_pool = concat_sequence_output.mean(1)

        if self.logical_info:
            new_punct_id = self.max_rel_id + 1   # new_punct_id:5
            new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 5. for incorporating with argument_bpe_ids.
            _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-4: arg, 5:punct.
            overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
            flat_all_bpe_ids = _flat_all_bpe_ids * (
                        1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
            assert flat_argument_bpe_ids.max().item() <= new_punct_id

            encoded_spans, span_mask, edges, edges_embed, node_in_seq_indices, attention_mask = self.split_into_spans_9(
                sequence_output,
                flat_attention_mask,
                flat_all_bpe_ids,
                flat_passage_mask,
                flat_option_mask,
                "coref")

            argument_graph, punctuation_graph, casual_graph, coref_graph = self.get_adjacency_matrices_2(
                edges, coref_tags=flat_coref_tags, n_nodes=encoded_spans.size(1), device=encoded_spans.device, type="coref")
            encoded_spans = encoded_spans + self.pos_embed(encoded_spans)  # node_embedding + positional embedding
            node = self.input_dropout(encoded_spans)
            coref_layer_output_list = []
            for enc_layer in self.encoder_layers:
                attn_bias = coref_graph
                # attn_bias = casual_graph
                node = enc_layer(node, attn_bias, attention_mask)
                coref_layer_output_list.append(node)
            node_coref = coref_layer_output_list[-1] + coref_layer_output_list[-2]
            node_coref = self.final_ln(node_coref)
            gcn_info_vec_coref = self.get_gcn_info_vector(node_in_seq_indices, node_coref,
                                                    size=sequence_output.size(), device=sequence_output.device)   # [batchsize*n_choice, seq_len, hidden_size]

            gcn_updated_sequence_output = self._enc(
                self._prj_ln(concat_sequence_output + gcn_info_vec_coref))
            
            sequence_h2_weight = self._proj_sequence_h(gcn_updated_sequence_output).squeeze(-1)
            passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
            passage_h2 = util.weighted_sum(gcn_updated_sequence_output, passage_h2_weight)
            option_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_option_mask.float())
            option_h2 = util.weighted_sum(gcn_updated_sequence_output, option_h2_weight)

            concat_pool_output = self.dropout(concat_pool)
            merged_feats = torch.cat([passage_h2, option_h2, gcn_updated_sequence_output[:, 0], concat_pool_output],
                                         dim=1)
            logits = self.classifier_merge(merged_feats)

        if self.two_subgraphs:
            new_punct_id = self.max_rel_id + 1   # new_punct_id:5
            new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 5. for incorporating with argument_bpe_ids.
            _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-4: arg, 5:punct.
            overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
            flat_all_bpe_ids = _flat_all_bpe_ids * (
                        1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
            assert flat_argument_bpe_ids.max().item() <= new_punct_id

            context_encoded_spans, context_edges, context_node_in_seq_indices, context_attention_mask, context_output, \
               qa_encoded_spans, qa_edges, qa_node_in_seq_indices, qa_attention_mask, qa_output = self.split_into_subgraphs(
                sequence_output,
                flat_attention_mask,
                flat_all_bpe_ids,
                flat_passage_mask,
                flat_option_mask,
                "coref")
            context_encoded_spans = torch.cat([context_encoded_spans, pooled_output.unsqueeze(1)], dim=1)
            qa_encoded_spans = torch.cat([qa_encoded_spans, pooled_output.unsqueeze(1)], dim=1)

            ''' update context graph features '''
            _, _, _, context_graph = self.get_adjacency_matrices_2(
                context_edges, coref_tags=flat_context_occ, n_nodes=context_encoded_spans.size(1), device=context_encoded_spans.device, type="coref")
            context_encoded_spans = context_encoded_spans + self.pos_embed(context_encoded_spans)  # node_embedding + positional embedding
            context_node = self.input_dropout(context_encoded_spans)
            coref_layer_output_list = []
            for enc_layer in self.encoder_layers:
                attn_bias = context_graph
                context_node = enc_layer(context_node, attn_bias, context_attention_mask)
                coref_layer_output_list.append(context_node)
            context_node = self.final_ln(coref_layer_output_list[-1] + coref_layer_output_list[-2])
            gcn_info_vec_context = self.get_gcn_info_vector(context_node_in_seq_indices, context_node,
                                                    size=context_output.size(), device=context_output.device)   # [batchsize*n_choice, seq_len, hidden_size]
            updated_context_output = self._enc_context(
                self._prj_ln(context_output + gcn_info_vec_context))

            ''' update qa graph features '''
            _, _, _, qa_graph = self.get_adjacency_matrices_2(
                qa_edges, coref_tags=flat_qa_occ, n_nodes=qa_encoded_spans.size(1), device=qa_encoded_spans.device, type="coref")

            qa_encoded_spans = qa_encoded_spans + self.pos_embed(qa_encoded_spans)  # node_embedding + positional embedding
            qa_node = self.input_dropout(qa_encoded_spans)
            coref_layer_output_list = []
            for enc_layer in self.encoder_layers:
                attn_bias = qa_graph
                qa_node = enc_layer(qa_node, attn_bias, qa_attention_mask)
                coref_layer_output_list.append(qa_node)
            qa_node = self.final_ln(coref_layer_output_list[-1] + coref_layer_output_list[-2])
            gcn_info_vec_qa = self.get_gcn_info_vector(qa_node_in_seq_indices, qa_node,
                                                    size=qa_output.size(), device=qa_output.device)   # [batchsize*n_choice, seq_len, hidden_size]
            updated_qa_output = self._enc_qa(
                self._prj_ln(qa_output + gcn_info_vec_qa))

            # sequence_h2_weight = self._proj_sequence_h(concat_sequence_output).squeeze(-1)
            # passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
            # passage_h2 = util.weighted_sum(concat_sequence_output, passage_h2_weight)
            # option_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_option_mask.float())
            # option_h2 = util.weighted_sum(concat_sequence_output, option_h2_weight)

            merged_feats = torch.cat([updated_context_output[:,0], updated_qa_output[:,0], concat_sequence_output[:,0], concat_pool], dim=1)
            # merged_feats = torch.cat([updated_context_output[:,0], updated_qa_output[:,0], concat_pool], dim=1)
            # output_contra = self.feedforward(merged_feats)
            output_contra = context_node[:,-1,:] + qa_node[:,-1,:]
            logits = self.classifier_merge(merged_feats)

        if self.one_graph:
            new_punct_id = self.max_rel_id + 1   # new_punct_id:5
            new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 5. for incorporating with argument_bpe_ids.
            _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-4: arg, 5:punct.
            overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
            flat_all_bpe_ids = _flat_all_bpe_ids * (
                        1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
            assert flat_argument_bpe_ids.max().item() <= new_punct_id

            context_encoded_spans, context_edges, context_node_in_seq_indices, attention_mask, context_output, \
               qa_encoded_spans, qa_edges, qa_node_in_seq_indices, attention_mask, qa_output = self.split_into_one_graph(
                sequence_output,
                flat_attention_mask,
                flat_all_bpe_ids,
                flat_passage_mask,
                flat_option_mask,
                "coref")
            encoded_spans = torch.cat([context_encoded_spans, qa_encoded_spans, pooled_output.unsqueeze(1)], dim=1)

            bias_graph = self.get_adjacency_matrices_one_graph(
                context_encoded_spans, context_coref_tags=flat_context_occ, qa_coref_tags=flat_qa_occ, n_nodes=encoded_spans.size(1), device=encoded_spans.device, type="coref")
            encoded_spans = encoded_spans + self.pos_embed(encoded_spans)  # node_embedding + positional embedding
            node = self.input_dropout(encoded_spans)
            coref_layer_output_list = []
            for enc_layer in self.encoder_layers:
                attn_bias = bias_graph
                node = enc_layer(node, attn_bias, attention_mask)
                coref_layer_output_list.append(node)
            node = self.final_ln(coref_layer_output_list[-1] + coref_layer_output_list[-2])

            sequence_h2_weight = self._proj_sequence_h(concat_sequence_output).squeeze(-1)
            passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
            passage_h2 = util.weighted_sum(concat_sequence_output, passage_h2_weight)
            option_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_option_mask.float())
            option_h2 = util.weighted_sum(concat_sequence_output, option_h2_weight)

            output_contra = node[:,-1,:]
            merged_feats = torch.cat([passage_h2, option_h2, output_contra, concat_pool], dim=1)
            logits = self.classifier_merge(merged_feats)

        # logits = self.classifier(concat_pool)  

        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)
        outputs = (reshaped_logits,)

        # two losses during training process #
        if labels is not None:
            device = sequence_output.device
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            if self.loss == "margin":
                flat_qtype = flat_qtype.repeat_interleave(4)
                self.des_embedding = self.des_embedding.to(device)   # [17, hidden_size]
                zeros = torch.tensor(0.).to(device)

                for a, b in enumerate(output_contra): 
                    max_val = torch.tensor(0.).to(device)
                    for i, j in enumerate(self.des_embedding):
                        if flat_qtype[a]==i:
                            pos = torch.dot(b, j.float()).to(device)
                        else:
                            tmp = torch.dot(b, j.float()).to(device)
                            if tmp > max_val:
                                max_val = tmp
                    neg = max_val.to(device)
                    loss += torch.max(zeros, neg - pos + self.gamma)*(1/pooled_output.size(0))*0.2

            elif self.loss == "InfoNCE":
                flat_qtype = flat_qtype.repeat_interleave(4)
                self.des_embedding = self.des_embedding.to(device)   # [17, hidden_size]

                for a, b in enumerate(concat_pool_output):
                    neg = torch.tensor(0.).to(device)
                    for i, j in enumerate(self.des_embedding):
                        if flat_qtype[a]==i:
                            pos = torch.exp(torch.dot(b, j.float())).to(device)
                        else:
                            neg += torch.exp(torch.dot(b, j.float())).to(device)
                    loss -= torch.log(pos/(pos+neg))*(1/pooled_output.size(0))*0.2
            outputs = (loss,) + outputs
        
        return outputs