# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from tools import allennlp as util
from transformers import BertPreTrainedModel, RobertaModel, BertModel, AlbertModel, XLNetModel, RobertaForMaskedLM, RobertaTokenizer
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

class qtype_Embedding(nn.Module):
    def __init__(self, hidden_size):
        super(qtype_Embedding, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, x):  # input is encoded spans
        batch_size = x.size(0)
        type_num = 19
        qtype_embed = torch.arange(type_num, dtype=torch.long)
        self.embedding = nn.Embedding(type_num, self.hidden_size)  # position embedding
        qtype_embed = self.embedding(qtype_embed)
        for i in range(batch_size):
            if i == 0:
                final_embed = qtype_embed[x[i].item(), :].unsqueeze(0)
            else:
                final_embed = torch.cat((final_embed, qtype_embed[x[i].item(), :].unsqueeze(0)), 0)
        # print(final_embed.size())
        return final_embed.to("cuda:0")


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


class DAGN_ques(BertPreTrainedModel):
    '''
    Adapted from https://github.com/llamazing/numnet_plus.

    Inputs of forward(): see try_data_5.py - the outputs of arg_tokenizer()
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}


    '''

    def __init__(self,
                 config,
                 init_weights: bool,
                 max_rel_id,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 merge_type: int = 1,
                 token_encoder_type: str = "roberta",
                 gnn_version: str = "GCN",
                 use_pool: bool = False,
                 use_gcn: bool = False,
                 gcn_steps: int = 1) -> None:
        super().__init__(config)

        self.layer_num = 5
        self.head_num = 5
        self.token_encoder_type = token_encoder_type
        self.max_rel_id = max_rel_id
        self.merge_type = merge_type
        self.use_gcn = use_gcn
        self.use_pool = use_pool
        assert self.use_gcn or self.use_pool

        ''' from modeling_roberta '''
        self.roberta = RobertaModel(config)

        if self.use_pool:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, 1)
        ''' from numnet '''
        if self.use_gcn:
            modeling_out_dim = hidden_size
            node_dim = modeling_out_dim

            self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
            if gnn_version == "GCN":
                self._gcn = ArgumentGCN(node_dim=node_dim, iteration_steps=gcn_steps)
            elif gnn_version == "GCN_reversededges_double":
                self._gcn = ArgumentGCN_wreverseedges_double(node_dim=node_dim, iteration_steps=gcn_steps)
            else:
                print("gnn_version == {}".format(gnn_version))
                raise Exception()
            self._iteration_steps = gcn_steps
            self._gcn_prj_ln = nn.LayerNorm(node_dim)
            # self._gcn_enc = ResidualGRU(hidden_size, 0, 2)
            self._gcn_enc = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.ReLU())
            # self._gru_enc = GRU_Feature_Fusion(hidden_size)

            self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)

            # span num extraction
            self._proj_span_num = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)
            # self._proj_span_num_2 = FFNLayer(2 * hidden_size, hidden_size, 1, dropout_prob)
            self._proj_gcn_pool = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)
            self._proj_gcn_pool_4 = FFNLayer(4 * hidden_size, hidden_size, 1, dropout_prob)
            self._proj_gcn_pool_3 = FFNLayer(2 * hidden_size, hidden_size, 1, dropout_prob)

            self.pre_ln = nn.LayerNorm(hidden_size)

            # self.qtype_embed = qtype_Embedding(hidden_size)
            self.pos_embed = Position_Embedding(hidden_size)

            self.input_dropout = nn.Dropout(dropout_prob)
            encoders = [EncoderLayer(hidden_size, hidden_size, dropout_prob, dropout_prob, self.head_num)
                        for _ in range(self.layer_num)]
            self.encoder_layers = nn.ModuleList(encoders)


            self.final_ln = nn.LayerNorm(hidden_size)
            # self.gate_params = FFNLayer(2*hidden_size, hidden_size, 2, 0.1)
            # self.gate_params = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU())
            # self.ques_attn = FFNLayer(2 * hidden_size, hidden_size, 1, dropout_prob)
            # self._proj_att_bias = FFNLayer(1, 512, 1, 0.1)

        if init_weights:
            self.init_weights()

    def split_into_spans_9(self, seq, seq_mask, split_bpe_ids, passage_mask, option_mask, question_mask, type):
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
        ques_seq = []
        for item_seq_mask, item_seq, item_split_ids, p_mask, o_mask, q_mask in zip(seq_mask, seq, split_bpe_ids,
                                                                                   passage_mask, option_mask,
                                                                                   question_mask):
            # item_seq_len = item_seq_mask.sum().item()
            item_seq_len = p_mask.sum().item() + o_mask.sum().item()  # item_seq = passage + option
            item_ques_seq = item_seq[item_seq_len:item_seq_mask.sum().item()]
            item_ques_seq = item_ques_seq.mean(dim=0)
            item_seq = item_seq[:item_seq_len]
            item_split_ids = item_split_ids[:item_seq_len]
            item_split_ids = item_split_ids.cpu().numpy()
            if type == "casual":
                split_ids_indices = np.where(item_split_ids > 0)[0].tolist()     # Casual type
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
            ques_seq.append(item_ques_seq)

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
        ques_seq = torch.stack(ques_seq, dim=0)

        return encoded_spans, span_masks, truncated_edges, truncated_edges_embed, node_in_seq_indices, attention_mask, ques_seq

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

            Note: relation patterns
                1 - (relation, head, tail)  关键词在句首
                2 - (head, relation, tail)  关键词在句中，先因后果
                3 - (tail, relation, head)  关键词在句中，先果后因
                4 - (head, relation, tail) & (tail, relation, head)  (1) argument words 中的一些关系
                5 - (head, relation, tail) & (tail, relation, head)  (2) punctuations

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
            # for i in range(coref_graph.size(1)):
            #     coref_graph[:, i, i] = 1
        return argument_graph.to(device), punct_graph.to(device), casual_graph.to(device), coref_graph.to(device)

        # return argument_graph.to(device), punct_graph.to(device), casual_graph.to(device)


    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,

                passage_mask: torch.LongTensor,
                option_mask: torch.LongTensor,
                question_mask: torch.LongTensor,

                argument_bpe_ids: torch.LongTensor,
                domain_bpe_ids: torch.LongTensor,
                punct_bpe_ids: torch.LongTensor,

                labels: torch.LongTensor,
                coref: torch.LongTensor,
                # qtype: torch.LongTensor,
                token_type_ids: torch.LongTensor = None,
                ) -> Tuple:
        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        flat_passage_mask = passage_mask.view(-1, passage_mask.size(
            -1)) if passage_mask is not None else None  # [num_choice*batchsize, hidden_size]
        flat_option_mask = option_mask.view(-1, option_mask.size(
            -1)) if option_mask is not None else None  # [num_choice*batchsize, hidden_size]
        flat_question_mask = question_mask.view(-1, question_mask.size(
            -1)) if question_mask is not None else None  # [num_choice*batchsize, hidden_size]

        flat_argument_bpe_ids = argument_bpe_ids.view(-1, argument_bpe_ids.size(
            -1)) if argument_bpe_ids is not None else None
        flat_domain_bpe_ids = domain_bpe_ids.view(-1, domain_bpe_ids.size(-1)) if domain_bpe_ids is not None else None
        flat_punct_bpe_ids = punct_bpe_ids.view(-1, punct_bpe_ids.size(-1)) if punct_bpe_ids is not None else None
        flat_coref_tags = coref.view(-1, coref.size(-2), coref.size(-1)) if coref is not None else None
        # flat_qtype = qtype.view(-1) if qtype is not None else None
        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]  # [bz*n_choice, hidden_size]

        if self.use_gcn:
            ''' The GCN branch. Suppose to go back to baseline once remove. '''
            new_punct_id = self.max_rel_id + 1   # new_punct_id:5
            new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 5. for incorporating with argument_bpe_ids.
            _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-4: arg, 5:punct.
            overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
            flat_all_bpe_ids = _flat_all_bpe_ids * (
                        1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
            assert flat_argument_bpe_ids.max().item() <= new_punct_id

            # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
            # span_mask: (bsz x n_choices, n_nodes)
            # edges: list[list[int]]
            # node_in_seq_indices: list[list[list[int]]]

            ''' Logical Casual '''

            encoded_spans, span_mask, edges, edges_embed, node_in_seq_indices, attention_mask, ques_seq = self.split_into_spans_9(
                sequence_output,
                flat_attention_mask,
                flat_all_bpe_ids,
                flat_passage_mask,
                flat_option_mask,
                flat_question_mask,
                "casual")
            argument_graph, punctuation_graph, casual_graph, coref_graph = self.get_adjacency_matrices_2(
                edges, coref_tags=flat_coref_tags, n_nodes=encoded_spans.size(1), device=encoded_spans.device, type="casual")
            encoded_spans = encoded_spans + self.pos_embed(encoded_spans)  # node_embedding + positional embedding

            node = self.input_dropout(encoded_spans)
            casual_layer_output_list = []
            for enc_layer in self.encoder_layers:
                # attn_bias = None
                attn_bias = casual_graph
                node = enc_layer(node, attn_bias, attention_mask)
                casual_layer_output_list.append(node)
            node_casual = casual_layer_output_list[-1] + casual_layer_output_list[-2]
            node_casual = self.final_ln(node_casual)
            gcn_info_vec_casual = self.get_gcn_info_vector(node_in_seq_indices, node_casual,
                                                    size=sequence_output.size(), device=sequence_output.device)


            ''' Co-reference Semantic '''

            encoded_spans, span_mask, edges, edges_embed, node_in_seq_indices, attention_mask, ques_seq1 = self.split_into_spans_9(
                sequence_output,
                flat_attention_mask,
                flat_all_bpe_ids,
                flat_passage_mask,
                flat_option_mask,
                flat_question_mask,
                "coref")
            argument_graph, punctuation_graph, casual_graph, coref_graph = self.get_adjacency_matrices_2(
                edges, coref_tags=flat_coref_tags, n_nodes=encoded_spans.size(1), device=encoded_spans.device, type="coref")
            encoded_spans = encoded_spans + self.pos_embed(encoded_spans)  # node_embedding + positional embedding
            node = self.input_dropout(encoded_spans)
            coref_layer_output_list = []
            for enc_layer in self.encoder_layers:
                # attn_bias = self._proj_att_bias(coref_graph.unsqueeze(-1)).squeeze(-1)
                attn_bias = coref_graph
                # attn_bias = None
                node = enc_layer(node, attn_bias, attention_mask)
                coref_layer_output_list.append(node)
            node_coref = coref_layer_output_list[-1] + coref_layer_output_list[-2]
            node_coref = self.final_ln(node_coref)
            gcn_info_vec_coref = self.get_gcn_info_vector(node_in_seq_indices, node_coref,
                                                    size=sequence_output.size(), device=sequence_output.device)   # [batchsize*n_choice, seq_len, hidden_size]



            ''' gate params for feature fusion '''
            '''
            gates = self.gate_params(torch.cat([gcn_info_vec_casual, gcn_info_vec_coref], dim=-1))
            gates = torch.softmax(gates, dim=-1)    # [batchsize*n_choice, seq_len, 2]

            gcn_updated_sequence_output = self._gcn_enc(
                 self._gcn_prj_ln(sequence_output + gates[:,:,0].unsqueeze(-1)*gcn_info_vec_coref + gates[:,:,1].unsqueeze(-1)*gcn_info_vec_casual))  # [batchsize*n_choice, seq_len, hidden_size]
            '''
            gcn_updated_sequence_output = self._gcn_enc(
                self._gcn_prj_ln(sequence_output + 0.6*gcn_info_vec_coref + 0.4*gcn_info_vec_casual))
            # gcn_updated_sequence_output = self._gru_enc(sequence_output, gcn_info_vec_casual, gcn_info_vec_coref)

            # passage hidden and question hidden
            sequence_h2_weight = self._proj_sequence_h(gcn_updated_sequence_output).squeeze(-1)
            passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
            passage_h2 = util.weighted_sum(gcn_updated_sequence_output, passage_h2_weight)
            question_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_question_mask.float())
            question_h2 = util.weighted_sum(gcn_updated_sequence_output, question_h2_weight)

            ''' get gcn logits '''
            gcn_output_feats = torch.cat([passage_h2, question_h2, gcn_updated_sequence_output[:, 0]], dim=1)
            gcn_logits = self._proj_span_num(gcn_output_feats)


        if self.use_pool:
            ''' The baseline branch. The output. '''
            # pooled_output = gcn_updated_sequence_output[:, 0]
            pooled_output = self.dropout(pooled_output)
            baseline_logits = self.classifier(pooled_output)

        if self.use_gcn and self.use_pool:
            ''' Merge gcn_logits & baseline_logits. TODO: different way of merging. '''

            if self.merge_type == 1:
                logits = gcn_logits + baseline_logits

            elif self.merge_type == 2:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([gcn_updated_sequence_output[:, 0], pooled_output], dim=1)
                logits = self._proj_gcn_pool_3(merged_feats)

            elif self.merge_type == 3:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([gcn_updated_sequence_output[:, 0], pooled_output,
                                          gcn_updated_sequence_output[:, 0], pooled_output], dim=1)
                logits = self._proj_gcn_pool_4(merged_feats)

            elif self.merge_type == 4:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([passage_h2, question_h2, pooled_output], dim=1)
                logits = self._proj_gcn_pool(merged_feats)
                logits = logits + 0.5*gcn_logits

            elif self.merge_type == 5:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([passage_h2, question_h2, gcn_updated_sequence_output[:, 0], pooled_output],
                                         dim=1)
                logits = self._proj_gcn_pool_4(merged_feats)

        elif self.use_gcn:
            logits = gcn_logits
        elif self.use_pool:
            logits = baseline_logits
        else:
            raise Exception


        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            outputs = (loss,) + outputs
        return outputs