from dataclasses import dataclass, field
import argparse
from transformers import AutoTokenizer
import re
import numpy as np
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from itertools import groupby
from operator import itemgetter
import copy
stemmer = SnowballStemmer("english")


def has_same_logical_component(set1, set2):
    has_same = False
    overlap = -1
    if len(set1) > 1 and len(set2) > 1:
        overlap = len(set1 & set2)/max(min(len(set1), len(set2)), 1)
        if overlap > 0.5:  # hyper-parameter:0.5
            has_same = 1
    return has_same, overlap


def token_stem(token):
    return stemmer.stem(token)

def get_node_tag(bpe_tokens):
    i = 0
    mask_tag, tag_now = 0, 0
    cond_tag, res_tag = 1, 2
    node_tag = []
    while i < len(bpe_tokens):
        if bpe_tokens[i] == "<cond>" or bpe_tokens[i] == "<mask>" or bpe_tokens[i] == "<unk>":
            tag_now += 1
            # node_tag.append(tag_now)
            if bpe_tokens[i] == "<mask>":
                node_tag.append(cond_tag)
            else:
                node_tag.append(res_tag)
            bpe_tokens.pop(i)
            i += 1
        elif bpe_tokens[i] == "</cond>" or bpe_tokens[i] == "</s>":
            bpe_tokens.pop(i)
        else:
            node_tag.append(mask_tag)
            i += 1
    return bpe_tokens, node_tag


def arg_tokenizer(text_a, text_b, text_c, tokenizer, max_length:int, do_lower_case:bool=False):
    '''
    :param text_a: str. (context in a sample.)
    :param text_b: str. ([#1] option in a sample. [#2] question + option in a sample.)
    :param text_c: str. (question in a sample)
    :param tokenizer: RoBERTa tokenizer.
    :param relations: dict. {argument words: pattern}
    :return:
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
    '''

    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)
    bpe_tokens_b = tokenizer.tokenize(text_b)
    bpe_tokens_c = tokenizer.tokenize(text_c)

    bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                    bpe_tokens_b + [tokenizer.sep_token] + \
                    bpe_tokens_c + [tokenizer.eos_token]

    a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
    b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - (len(bpe_tokens_a) + 2 + len(bpe_tokens_b) + 1))
    c_mask = [0] * (len(bpe_tokens_a) + 2) + [0] * (len(bpe_tokens_b) + 1) + [1] * (len(bpe_tokens_c) + 1) + [0] * (max_length - len(bpe_tokens))

    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    c_mask = c_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)
    assert len(c_mask) == max_length, 'len_c_mask={}, max_len={}'.format(len(c_mask), max_length)


    ''' output items '''
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)
    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))
    input_ids += padding_ids
    input_mask += padding
    segment_ids += padding

    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)


    output = {}
    output["input_tokens"] = bpe_tokens
    output["input_ids"] = input_ids
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids

    return output


def prompt_tokenizer(prefix, qtype, text_a, text_b, text_c, tokenizer, stopwords, relations:dict, punctuations:list, max_gram:int, max_length:int, do_lower_case:bool=False):

    '''
    :param text_a: str. (context in a sample.)
    :param text_b: str. (question in a sample, which may contain "_" for the prompt)
    :param text_c: str. (option in a sample)
    :param tokenizer: RoBERTa tokenizer.
    :param relations: dict. {argument words: pattern}
    :return:
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
    '''

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _is_stopwords(word, stopwords):
        if word in stopwords:
            is_stopwords_flag = True
        else:
            is_stopwords_flag = False
        return is_stopwords_flag

    def _head_tail_is_stopwords(span, stopwords):
        if span[0] in stopwords or span[-1] in stopwords:
            return True
        else:
            return False

    def _with_septoken(ngram, tokenizer):
        if tokenizer.bos_token in ngram or tokenizer.sep_token in ngram or tokenizer.eos_token in ngram:
            flag = True
        else: flag = False
        return flag

    def _is_argument_words(seq, argument_words):
        pattern = None
        arg_words = list(argument_words.keys())
        if seq.strip() in arg_words:
            pattern = argument_words[seq.strip()]
        return pattern

    def _is_exist(exists:list, start:int, end:int):
        flag = False
        for estart, eend in exists:
            if estart <= start and eend >= end:
                flag = True
                break
        return flag

    def _find_punct(tokens, punctuations):
        punct_ids = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in punctuations:
                punct_ids[i] = 1
        return punct_ids

    def _find_arg_ngrams(tokens, max_gram):
        n_tokens = len(tokens)
        global_arg_start_end = []
        argument_words = {}
        argument_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                ngram = " ".join(tokens[window_start:window_end])
                pattern = _is_argument_words(ngram, relations)
                if pattern:
                    if not _is_exist(global_arg_start_end, window_start, window_end):
                        global_arg_start_end.append((window_start, window_end))
                        argument_ids[window_start:window_end] = [pattern] * (window_end - window_start)
                        argument_words[ngram] = (window_start, window_end)

        return argument_words, argument_ids

    def _find_dom_ngrams_2(tokens, max_gram):
        '''
        1. 判断 stopwords 和 sep token
        2. 先遍历一遍，记录 n-gram 的重复次数和出现位置
        3. 遍历记录的 n-gram, 过滤掉 n-gram 子序列（直接比较 str）
        4. 赋值 domain_ids.

        '''

        stemmed_tokens = [token_stem(token) for token in tokens]

        ''' 1 & 2'''
        n_tokens = len(tokens)
        d_ngram = {}
        domain_words_stemmed = {}
        domain_words_orin = {}
        domain_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.

                window_start, window_end = i, i+n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = tokens[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if _is_stopwords(orin_ngram, stopwords): continue
                if _head_tail_is_stopwords(orin_span, stopwords): continue
                if _with_septoken(orin_ngram, tokenizer): continue

                if not stemmed_ngram in d_ngram:
                    d_ngram[stemmed_ngram] = []
                d_ngram[stemmed_ngram].append((window_start, window_end))

        ''' 3 '''
        d_ngram = dict(filter(lambda e: len(e[1]) > 1, d_ngram.items()))
        raw_domain_words = list(d_ngram.keys())
        raw_domain_words.sort(key=lambda s: len(s), reverse=True)  # sort by len(str).
        domain_words_to_remove = []
        for i in range(0, len(d_ngram)):
            for j in range(i+1, len(d_ngram)):
                if raw_domain_words[i] in raw_domain_words[j]:
                    domain_words_to_remove.append(raw_domain_words[i])
                if raw_domain_words[j] in raw_domain_words[i]:
                    domain_words_to_remove.append(raw_domain_words[j])
        for r in domain_words_to_remove:
            try:
                del d_ngram[r]
            except:
                pass

        ''' 4 '''
        d_id = 0
        for stemmed_ngram, start_end_list in d_ngram.items():
            d_id += 1
            for start, end in start_end_list:
                domain_ids[start:end] = [d_id] * (end - start)
                rebuilt_orin_ngram = " ".join(tokens[start: end])
                if not stemmed_ngram in domain_words_stemmed:
                    domain_words_stemmed[stemmed_ngram] = []
                if not rebuilt_orin_ngram in domain_words_orin:
                    domain_words_orin[rebuilt_orin_ngram] = []
                domain_words_stemmed[stemmed_ngram] +=  [(start, end)]
                domain_words_orin[rebuilt_orin_ngram] += [(start, end)]


        return domain_words_stemmed, domain_words_orin, domain_ids



    ''' fill in the prompt'''
    if text_b.find('_') != -1:
        text_b = text_b.replace('_', text_c)
    else:
        text_b = text_b + ' ' + text_c

    ''' convert questions into statements '''
    if text_b.find('?') != -1:
        text_b = text_b.replace('?', '.')


    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)
    bpe_tokens_b = tokenizer.tokenize(text_b)

    # get bpe tokens
    if prefix is not None:
        bpe_tokens_prefix = tokenizer.tokenize(prefix)
        bpe_tokens = [tokenizer.bos_token] + bpe_tokens_prefix + [tokenizer.sep_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                        bpe_tokens_b + [tokenizer.eos_token]
        a_mask = [1] * (len(bpe_tokens_a) + len(bpe_tokens_prefix) + 3) + [0] * (max_length - (len(bpe_tokens_a) + len(bpe_tokens_prefix) + 3))
        b_mask = [0] * (len(bpe_tokens_a) + len(bpe_tokens_prefix) + 3) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - (len(bpe_tokens_a) + len(bpe_tokens_prefix) + 3 + len(bpe_tokens_b) + 1))

    else:
        bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                        bpe_tokens_b + [tokenizer.eos_token]
        a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
        b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - (len(bpe_tokens_a) + 2 + len(bpe_tokens_b) + 1))

    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)

    # adapting Ġ.
    assert isinstance(bpe_tokens, list)
    bare_tokens = [token[1:] if "Ġ" in token else token for token in bpe_tokens]
    argument_words, argument_space_ids = _find_arg_ngrams(bare_tokens, max_gram=max_gram)
    domain_words_stemmed, domain_words_orin, domain_space_ids = _find_dom_ngrams_2(bare_tokens, max_gram=max_gram)
    punct_space_ids = _find_punct(bare_tokens, punctuations)
    argument_bpe_ids = argument_space_ids
    domain_bpe_ids = domain_space_ids
    punct_bpe_ids = punct_space_ids

    ''' output items '''
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
    input_mask = [1] * len(input_ids)
    if prefix is not None:
        segment_ids = [2] * (len(bpe_tokens_prefix) + 2) + [0] * (len(bpe_tokens_a) + 1) + [1] * (len(bpe_tokens_b) + 1)
    else:
        segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)
    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))
    input_ids += padding_ids
    argument_bpe_ids += arg_dom_padding_ids
    domain_bpe_ids += arg_dom_padding_ids
    punct_bpe_ids += arg_dom_padding_ids
    input_mask += padding
    segment_ids += padding

    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]
    argument_bpe_ids = argument_bpe_ids[:max_length]
    domain_bpe_ids = domain_bpe_ids[:max_length]
    punct_bpe_ids = punct_bpe_ids[:max_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)
    assert len(argument_bpe_ids) <= max_length, 'len_argument_bpe_ids={}, max_length={}'.format(
        len(argument_bpe_ids), max_length)
    assert len(domain_bpe_ids) <= max_length, 'len_domain_bpe_ids={}, max_length={}'.format(
        len(domain_bpe_ids), max_length)
    assert len(punct_bpe_ids) <= max_length, 'len_punct_bpe_ids={}, max_length={}'.format(
        len(punct_bpe_ids), max_length)


    ''' get the co-reference relation new version '''
    ''' Copy From Logiformer '''
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

    max_rel_id = 4
    new_punct_id = max_rel_id + 1  # new_punct_id:5
    new_punct_bpe_ids = [i*new_punct_id for i in punct_bpe_ids]  # punct_id: 1 -> 5. for incorporating with argument_bpe_ids.
    _flat_all_bpe_ids = list(map(lambda x,y:x+y, argument_bpe_ids, new_punct_bpe_ids))  # -1:padding, 0:non, 1-4: arg, 5:punct.
    overlapped_punct_argument_mask = [1 if bpe_id > new_punct_id else 0 for bpe_id in _flat_all_bpe_ids]
    flat_all_bpe_ids = list(map(lambda x,y:x*y, _flat_all_bpe_ids, [1-i for i in overlapped_punct_argument_mask])) \
                        + list(map(lambda x,y:x*y, argument_bpe_ids, overlapped_punct_argument_mask))
    assert max(argument_bpe_ids) <= new_punct_id


    ''' co-occurrence for context graph '''
    item_split_ids = np.array(flat_all_bpe_ids[:sum(a_mask)])  # type:numpy.array,
    split_ids_indices = np.where(item_split_ids > 0)[0].tolist()  # select id==5(punctuation)
    grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
        split_ids_indices, item_split_ids)
    # print(split_ids_indices)  # [0, 16, 20, 26, 30, 44, 53, 63, 71, 85, 93, 112]
    n_split_ids = len(split_ids_indices)    # the number of split_ids
    item_node_in_seq_indices = []
    sent_list = []
    for i in range(n_split_ids):
        if i != n_split_ids - 1:
            item_node_in_seq_indices.append([i for i in range(grouped_split_ids_indices[i][-1] + 1,
                                                              grouped_split_ids_indices[i + 1][0])])
            sent_list.append(bare_tokens[item_node_in_seq_indices[-1][0]:item_node_in_seq_indices[-1][-1]+1])
    sent_token_set = [set(sent) - set(stopwords+["<", ">", "b", "i", "e", "g", "</", "."]) for sent in sent_list]  # delete the stopwords and convert to set
    context_occ = []    # initialize the coref matrix

    max_nodes = 128
    for i in range(len(sent_token_set)):
        for j in range(i+1, len(sent_token_set)):
            has_same, overlap = has_same_logical_component(sent_token_set[i], sent_token_set[j])
            if has_same:    # judge has_same
                context_occ.append((i, j, overlap))
    context_occ += [(-1,-1,-1)] * (max_nodes-len(context_occ))
    assert len(context_occ) <= max_nodes, 'len_context_occ={}, max_nodes={}'.format(
        len(context_occ), max_nodes)

    ''' co-occurrence for qa graph '''
    item_split_ids = np.array(flat_all_bpe_ids[sum(a_mask):sum(a_mask)+sum(b_mask)])  # type:numpy.array,
    split_ids_indices = np.where(item_split_ids > 0)[0].tolist()  # select id==5(punctuation)
    grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
        split_ids_indices, item_split_ids)
    # print(split_ids_indices)  # [0, 16, 20, 26, 30, 44, 53, 63, 71, 85, 93, 112]
    n_split_ids = len(split_ids_indices)    # the number of split_ids
    item_node_in_seq_indices = []
    sent_list = []
    for i in range(n_split_ids):
        if i != n_split_ids - 1:
            item_node_in_seq_indices.append([i for i in range(grouped_split_ids_indices[i][-1] + 1,
                                                              grouped_split_ids_indices[i + 1][0])])
            sent_list.append(bare_tokens[item_node_in_seq_indices[-1][0]:item_node_in_seq_indices[-1][-1]+1])
    sent_token_set = [set(sent) - set(stopwords+["<", ">", "b", "i", "e", "g", "</", "."]) for sent in sent_list]  # delete the stopwords and convert to set
    qa_occ = []    # initialize the coref matrix

    max_nodes = 128
    for i in range(len(sent_token_set)):
        for j in range(i+1, len(sent_token_set)):
            has_same, overlap = has_same_logical_component(sent_token_set[i], sent_token_set[j])
            if has_same:    # judge has_same
                qa_occ.append((i, j, overlap))
    qa_occ += [(-1,-1,-1)] * (max_nodes-len(qa_occ))
    assert len(qa_occ) <= max_nodes, 'len_qa_occ={}, max_nodes={}'.format(
        len(qa_occ), max_nodes)

    output = {}
    output["input_tokens"] = bpe_tokens
    output["input_ids"] = input_ids
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids
    output["argument_bpe_ids"] = argument_bpe_ids
    output["domain_bpe_ids"] = domain_bpe_ids
    output["punct_bpe_ids"] = punct_bpe_ids
    output["a_mask"] = a_mask
    output["b_mask"] = b_mask
    output["context_occ"] = context_occ
    output["qa_occ"] = qa_occ

    return output
