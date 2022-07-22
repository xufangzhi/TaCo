import numpy as np
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import jsonlines
import gensim
import tqdm
from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)

""" train split distribution for types """
type_distribution = [0.10845191893057353,
                    0.03794739111686071,
                    0.08753773178094006,
                    0.1034928848641656,
                    0.017033203967227253,
                    0.06295817162570073,
                    0.032772746873652434,
                    0.0646830530401035,
                    0.08106942647692972,
                    0.0579991375592928,
                    0.02457956015523933,
                    0.03708495040965933,
                    0.036653730056058646,
                    0.11621388529538594,
                    0.04678740836567486,
                    0.0258732212160414,
                    0.05886157826649418]

acc_distribution = [0.10845191893057353,
                    0.14639931004743423,
                    0.23393704182837427,
                    0.33742992669253985,
                    0.3544631306597671,
                    0.4174213022854678,
                    0.45019404915912026,
                    0.5148771021992238,
                    0.5959465286761535,
                    0.6539456662354463,
                    0.6785252263906856,
                    0.7156101768003449,
                    0.7522639068564035,
                    0.8684777921517894,
                    0.9152652005174643,
                    0.9411384217335057,
                    0.9999999999999999]

train_type_num = [503,176,406,480,79,292,152,300,376,269,114,172,170,539,217,120,273]

type2describe_prefix = {
    0:"This is the task of Necessary Assumptions.",
    1:"This is the task of Sufficient Assumptions.",
    2:"This is the task of Strengthen.",
    3:"This is the task of Weaken.",
    4:"This is the task of Evaluation.",
    5:"This is the task of Implication.",
    6:"This is the task of Conclusion and Main Point.",
    7:"This is the task of Most Strongly Supported.",
    8:"This is the task of Explain or Resolve.",
    9:"This is the task of Principle.",
    10:"This is the task of Dispute.",
    11:"This is the task of Technique.",
    12:"This is the task of Role.",
    13:"This is the task of Identify a Flaw.",
    14:"This is the task of Match Flaws.",
    15:"This is the task of Match the Structure.",
}

type2describe = {
    0:"identify the claim that must be true or is required in order for the argument to work.",
    1:"identify a sufficient assumption, that is, an assumption that, if added to the argument, would make it logically valid.",
    2:"identify information that would strengthen an argument.",
    3:"identify information that would weaken an argument.",
    4:"identify information that would be useful to know to evaluate an argument.",
    5:"identify something that follows logically from a set of premises.",
    6:"identify the conclusion/main point of a line of reasoning.",
    7:"find the choice that is most strongly supported by a stimulus.",
    8:"identify information that would explain or resolve a situation.",
    9:"identify the principle, or find a situation that conforms to a principle, or match the principles.",
    10:"identify or infer an issue in dispute.",
    11:"identify the technique used in the reasoning of an argument.",
    12:"describe the individual role that a statement is playing in a larger argument.",
    13:"identify a flaw in an arguments reasoning.",
    14:"find a choice containing an argument that exhibits the same flaws as the passages argument.",
    15:"match the structure of an argument in a choice to the structure of the argument in the passage.",
    16:"other types of questions which are not included by the above.",
}

@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]
    qtype: Optional[int]


@dataclass(frozen=True)
class InputFeatures:
    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    passage_mask: Optional[List[int]]
    option_mask: Optional[List[int]]
    argument_bpe_ids: Optional[List[List[int]]]
    domain_bpe_ids: Optional[List[List[int]]]
    punct_bpe_ids: Optional[List[List[int]]]
    label: Optional[int]
    context_occ: Optional[List[List[tuple]]]
    qa_occ: Optional[List[List[tuple]]]
    qtype: Optional[int]

class Split(Enum):
    train = "train"
    dev = "eval"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MyMultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            arg_tokenizer,
            relations,
            punctuations,
            task: str,
            max_seq_length: Optional[int] = None,
            max_ngram: int = 5,
            overwrite_cache=False,
            mode: Split = Split.train,
            sample_strategy=None,
        ):
            processor = processors[task]()

            if not os.path.isdir(os.path.join(data_dir, "cached_data")):
                os.mkdir(os.path.join(data_dir, "cached_data"))

            cached_features_file = os.path.join(
                data_dir,
                "cached_data",
                "dagn_cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    elif mode == Split.train:
                        if sample_strategy:
                            examples = processor.get_train_samples(data_dir, sample_strategy)
                        else:
                            examples = processor.get_train_examples(data_dir)
                    else:
                        raise Exception()
                    logger.info("Training examples: %s", len(examples))


                    self.features = convert_examples_to_arg_features(
                        examples,
                        label_list,
                        arg_tokenizer,
                        relations,
                        punctuations,
                        max_seq_length,
                        tokenizer,
                        max_ngram,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()



class ReclorProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train_qtype.json")), "train")

    def get_train_demos(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "100_train.json")), "train")

    def get_train_samples(self, data_dir, strategy=None):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_sample_examples(self._read_json(os.path.join(data_dir, "train_qtype.json")), "train", strategy)
        

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val_qtype.json")), "dev")

    def get_dev_demos(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "100_val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test_qtype.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _read_jsonl(self, input_file):
        reader = jsonlines.Reader(open(input_file, "r"))
        lines = [each for each in reader]
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label'] # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            qtype = d['qtype']
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label,
                    qtype = qtype
                    )
                )
        return examples
    
    def _create_sample_examples(self, lines, type, strategy=None):
        """Sample examples for the training based on different strategies"""
        """
            strategies:
                [1] random: randomly sample certain number of examples, without considering question type.
                [2] cut_one_type: delete one type of questions during training.
                [3] cut_some_types: delete some types of questions during training.
                [4] select_some_types: select some types of questions during training.
        """
        examples = []
        if strategy == "random":   # [1] random
            sample_number = 2190
            while len(examples)<sample_number:
                for d in lines:
                    context = d['context']
                    question = d['question']
                    answers = d['answers']
                    label = d['label']
                    id_string = d['id_string']
                    qtype = d['qtype']
                    ratio = sample_number / 4638
                    if len(examples) == sample_number:
                        break
                    if np.random.random(1)<=ratio:
                        examples.append(
                            InputExample(
                                example_id = id_string,
                                question = question,
                                contexts=[context, context, context, context],  # this is not efficient but convenient
                                endings=[answers[0], answers[1], answers[2], answers[3]],
                                label = label,
                                )
                            )
        else:
            for d in lines:
                context = d['context']
                question = d['question']
                answers = d['answers']
                label = d['label']
                id_string = d['id_string']
                qtype = d['qtype']

                if strategy == "cut_one_type":         # [2] cut_one_type
                    type_id = 0
                    if qtype != type_id:
                        examples.append(
                            InputExample(
                                example_id = id_string,
                                question = question,
                                contexts=[context, context, context, context],  # this is not efficient but convenient
                                endings=[answers[0], answers[1], answers[2], answers[3]],
                                label = label,
                                )
                            )
                elif strategy == "cut_some_types":         # [3] cut_some_types
                    v1 = [0,1,2,3,4]
                    v2 = [12,13,14,15,16]
                    if qtype not in v1:
                        examples.append(
                            InputExample(
                                example_id = id_string,
                                question = question,
                                contexts=[context, context, context, context],  # this is not efficient but convenient
                                endings=[answers[0], answers[1], answers[2], answers[3]],
                                label = label,
                                )
                            )

                elif strategy == "select_some_types":         # [4] select_some_types
                    v1 = [0]
                    v2 = [0,2,3,7,13]
                    v3 = [0,2,3,5,7,8,13]
                    v4 = [0,1,2,3,8,9,14,16]
                    v5 = [0,3,5,8,13]
                    if qtype in v5:
                        examples.append(
                            InputExample(
                                example_id = id_string,
                                question = question,
                                contexts=[context, context, context, context],  # this is not efficient but convenient
                                endings=[answers[0], answers[1], answers[2], answers[3]],
                                label = label,
                                qtype = qtype,
                                )
                            )
                else:
                    raise Exception("The sample strategy is not correct")
            
        logger.info("In total {} traing samples".format(len(examples)))
        return examples



class LogiQAProcessor(DataProcessor):
    """ Processor for the LogiQA data set. """

    def get_demo_examples(self, data_dir):
        logger.info("LOOKING AT {} demo".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "10_logiqa.txt")), "demo")

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Train.txt")), "train")

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Eval.txt")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Test.txt")), "test")

    def get_labels(self):
        return [0, 1, 2, 3]

    def _read_txt(self, input_file):
        with open(input_file, "r") as f:
            lines = f.readlines()
        return lines

    def _create_examples(self, lines, type):
        """ LogiQA: each 8 lines is one data point.
                The first line is blank line;
                The second is right choice;
                The third is context;
                The fourth is question;
                The remaining four lines are four options.
        """
        label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        assert len(lines) % 8 ==0, 'len(lines)={}'.format(len(lines))
        n_examples = int(len(lines) / 8)
        examples = []
        # for i, line in enumerate(examples):
        for i in range(n_examples):
            label_str = lines[i*8+1].strip()
            context = lines[i*8+2].strip()
            question = lines[i*8+3].strip()
            answers = lines[i*8+4 : i*8+8]

            examples.append(
                InputExample(
                    example_id = " ",  # no example_id in LogiQA.
                    question = question,
                    contexts = [context, context, context, context],
                    endings = [item.strip()[2:] for item in answers],
                    label = label_map[label_str]
                )
            )
        assert len(examples) == n_examples
        return examples


def manual_prompt(raw_question):
    trigger = ['which one of the following', 'Which one of the following', 'Which of the following', 
               'which of the following', 'Each of the following', 'Of the following, which one',
              'Each one of the following statements','which of the of the following',
               'Each one of the following','each of the following',
               'Which if the following', 'The following', 'Any of the following statements',
               'Which only of the following', 'Which one the following',
               'Of the following claims, which one', 'Any of the following',
               'All of the following', 'Which of following',
               'Of the following statements, which one','which one of me following',
               'Of the following claims, which', 'Of the following propositions, which one',
               'Which one of me following', 'Which of he following'
              ]
    for t in trigger:
        if raw_question.find(t) != -1:
            return raw_question.replace(t, "_")
    return raw_question


def convert_examples_to_arg_features(
    examples: List[InputExample],
    label_list: List[str],
    arg_tokenizer,
    relations: Dict,
    punctuations: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    max_ngram: int,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`

    context -> chunks of context
            -> domain_words to Dids
    option -> chunk of option
           -> domain_words in Dids
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            if example.qtype in type2describe_prefix.keys():
                prefix = type2describe_prefix[example.qtype]
            else:
                prefix = None
            text_a = context
            text_b = manual_prompt(example.question)   # question containing _
            # text_b = example.question
            text_c = ending

            stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations
            inputs = arg_tokenizer(prefix, example.qtype, text_a, text_b, text_c, tokenizer, stopwords, relations, punctuations, max_ngram, max_length)
            choices_inputs.append(inputs)

        label = label_map[example.label]
        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )

        a_mask = [x["a_mask"] for x in choices_inputs]
        b_mask = [x["b_mask"] for x in choices_inputs]  # list[list]
        argument_bpe_ids = [x["argument_bpe_ids"] for x in choices_inputs]
        if isinstance(argument_bpe_ids[0], tuple):  # (argument_bpe_pattern_ids, argument_bpe_type_ids)
            arg_bpe_pattern_ids, arg_bpe_type_ids = [], []
            for choice_pattern, choice_type in argument_bpe_ids:
                assert (np.array(choice_pattern) > 0).tolist() == (np.array(choice_type) > 0).tolist(), 'pattern: {}\ntype: {}'.format(
                    choice_pattern, choice_type)
                arg_bpe_pattern_ids.append(choice_pattern)
                arg_bpe_type_ids.append(choice_type)
            argument_bpe_ids = (arg_bpe_pattern_ids, arg_bpe_type_ids)
        domain_bpe_ids = [x["domain_bpe_ids"] for x in choices_inputs]
        punct_bpe_ids = [x["punct_bpe_ids"] for x in choices_inputs]
        # coref_tags = [x["coref"] for x in choices_inputs]
        context_occ = [x["context_occ"] for x in choices_inputs]
        qa_occ = [x["qa_occ"] for x in choices_inputs]

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                passage_mask=a_mask,
                option_mask=b_mask,
                argument_bpe_ids=argument_bpe_ids,
                domain_bpe_ids=domain_bpe_ids,
                punct_bpe_ids=punct_bpe_ids,
                label=label,
                context_occ=context_occ,
                qa_occ=qa_occ,
                qtype=example.qtype,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features

processors = {"reclor": ReclorProcessor, "logiqa": LogiQAProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag", 4, "arc", 4, "syn", 5, "reclor", 4, "logiqa", 4}
