import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk 
import numpy as np
from datasets import Dataset

from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.file_utils import is_offline_mode

START_DESCRIPTION = '<desc>'
END_DESCRIPTION = '</desc>'
START_COMMIT = '<cmt>'
END_COMMIT = '</cmt>'
START_ISSUE = '<iss>'
END_ISSUE = '</iss>'

EXTRA_TOKENS = [
    START_DESCRIPTION,
    END_DESCRIPTION,
    START_COMMIT,
    END_COMMIT,
    START_ISSUE,
    END_ISSUE
]

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )

class Model:
    def __init__(self, model_path):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        self.model_args, self.data_args, self.training_args = parser.parse_json_file(json_file='title_generator/config.json')
        print(f"model_args: {self.model_args}")
        print(f"data_args: {self.data_args}")
        print(f"training_args: {self.training_args}")

        config = AutoConfig.from_pretrained(
        self.model_args.config_name if self.model_args.config_name else model_path,
        cache_dir=self.model_args.cache_dir,
        revision=self.model_args.model_revision,
        use_auth_token=True if self.model_args.use_auth_token else None,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name if self.model_args.tokenizer_name else model_path,
        cache_dir=self.model_args.cache_dir,
        use_fast=self.model_args.use_fast_tokenizer,
        revision=self.model_args.model_revision,
        use_auth_token=True if self.model_args.use_auth_token else None,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        self.padding = "max_length" if self.data_args.pad_to_max_length else False

        self.prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(self.tokenizer, MBartTokenizer):
                self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.data_args.lang]
            else:
                self.model.config.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.data_args.lang)
        
        if (hasattr(self.model.config, "max_position_embeddings") and self.model.config.max_position_embeddings < self.data_args.max_source_length):
            if self.model_args.resize_position_embeddings is None:
                logger.warning(
                    f"Increasing the model's number of position embedding vectors from {self.model.config.max_position_embeddings} "
                    f"to {self.data_args.max_source_length}."
                )
                self.model.resize_position_embeddings(self.data_args.max_source_length)
            elif self.model_args.resize_position_embeddings:
                self.model.resize_position_embeddings(self.data_args.max_source_length)

        # Data collator
        self.label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=self.label_pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

         # Initialize our Trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=None,
        )

    def preprocess_text(self, text):
            inputs = [text]
            inputs = [self.prefix + str(inp) for inp in inputs]
            model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)
            return Dataset.from_dict(model_inputs)

    def predict(self, text):
        predict_dataset = self.preprocess_text(text)
        max_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )
        num_beams = self.data_args.num_beams if self.data_args.num_beams is not None else self.training_args.generation_num_beams
        print(f"predict_dataset: {predict_dataset}")
        predict_results = self.trainer.predict(
            predict_dataset,
            max_length=max_length, num_beams=num_beams
        )
        predictions = self.tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        return predictions[0]
        
model = Model('/app/model/checkpoint-66500')
def get_model():
    return model

if __name__ == "__main__":
    model = Model('/app/model/checkpoint-66500')
    print(model.predict("add readme and fix some classes not found error"))



