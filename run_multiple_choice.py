import torch
import logging
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer, AdamW,
    TrainingArguments,
    set_seed,
    RobertaModel,
    RobertaForMultipleChoice,
)
from utils_multiple_choice import processors
from collections import Counter
from tokenization import arg_tokenizer, prompt_tokenizer
from utils_multiple_choice import Split, MyMultipleChoiceDataset
from ZsLR import ZsLR

from graph_building_blocks.argument_set_punctuation_v4 import punctuations
with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
    relations = json.load(f)  # key: relations, value: ignore

logger = logging.getLogger(__name__)

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

def generate_type_embed(type2describe:dict):
    from sentence_transformers import SentenceTransformer

    description_list = [type2describe[i] for i in range(len(type2describe))]

    encoder = SentenceTransformer('bert-large-nli-mean-tokens')
    sentence_embeddings = torch.Tensor(encoder.encode(description_list))
    # sentence_embeddings = sentence_embeddings.mean(dim=-1)
    return sentence_embeddings

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

@dataclass
class ModelArguments:
    """
    Arguments pretaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    attention_drop: float = field(
        default=0.1,
        metadata={"help": "huggingface RoBERTa config.attention_probs_dropout_prob"}
    )
    hidden_drop: float = field(
        default=0.1,
        metadata={"help": "huggingface RoBERTa config.hidden_dropout_prob"}
    )
    init_weights: bool = field(
        default=False,
        metadata={"help": "init weights in Argument NumNet."}
    )
    # training
    roberta_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating roberta parameters"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pretaining to what data we are going to input our model for training and eval.
    """
    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    data_type: str = field(
        default="argument_numnet",
        metadata={
            "help": "data types in utils script. roberta_large | argument_numnet "
        }
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    sample_strategy: str = field(
        default=None,
        metadata={"help": "strategies for sample training data"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    sentence_embeddings = generate_type_embed(type2describe)    # The question type description embeddings [17,1024]

    # model = RobertaForMultipleChoice.from_pretrained(model_args.model_name_or_path)
    model = ZsLR.from_pretrained(
        model_args.model_name_or_path,
        # "/home/linqika/xufangzhi/ZsLR/checkpoints/reclor/ZsLR_occ/",
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        des_embedding=sentence_embeddings,
    )
    print(sum(x.numel() for x in model.parameters()))
    train_dataset = (
        MyMultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            arg_tokenizer=prompt_tokenizer,
            relations=relations,
            punctuations=punctuations,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            sample_strategy=data_args.sample_strategy,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MyMultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            arg_tokenizer=prompt_tokenizer,
            relations=relations,
            punctuations=punctuations,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        MyMultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            arg_tokenizer=prompt_tokenizer,
            relations=relations,
            punctuations=punctuations,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        if training_args.do_predict
        else None
        )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("roberta")
                        and not any(nd in n for nd in no_decay)],
            "lr": model_args.roberta_lr,
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("roberta")
                        and any(nd in n for nd in no_decay)],
            "lr": model_args.roberta_lr,
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
    )


    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None)
        )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.mode_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

            results.update(result)

        eval_result = trainer.predict(eval_dataset)
        preds = eval_result.predictions  # np array. (1000, 4)
        pred_ids = np.argmax(preds, axis=1)

        output_test_file = os.path.join(training_args.output_dir, "eval_predictions.npy")
        np.save(output_test_file, pred_ids)
        logger.info("predictions saved to {}".format(output_test_file))

    # Test
    if training_args.do_predict:
        if data_args.task_name == "reclor":
            logger.info("*** Test ***")

            test_result = trainer.predict(test_dataset)
            preds = test_result.predictions  # np array. (1000, 4)
            pred_ids = np.argmax(preds, axis=1)

            output_test_file = os.path.join(training_args.output_dir, "predictions.npy")
            np.save(output_test_file, pred_ids)
            logger.info("predictions saved to {}".format(output_test_file))
        elif data_args.task_name == "logiqa":
            logger.info("*** Test ***")

            test_result = trainer.predict(test_dataset)

            output_test_file = os.path.join(training_args.output_dir, "test_results.txt")

            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in test_result.metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(test_result.metrics)

if __name__ == "__main__":
    main()