---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:41261
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Systems: twine, inform, choicescript. Tags: parser, fantasy, female protagonist, horror, science fiction, multiple endings, choice-based, short, male protagonist, humor, graphics, gender-neutral protagonist, surreal, slice of life, mystery, second person, score, choice of games, sound, built-in hints', "Title: The Box. Author: Paul Michael Winters. Systems: kreate. Tags: escape room, parser, puzzles, second person, present tense, escape, collector, puzzle box. Description: You are a collector of rare antiquities, with a special passion for puzzle boxes. You've been searching for one particular box for a long time, but perhaps this box is better left unfound."],
    ['Systems: inform, tads, zil. Tags: parser, male protagonist, slice of life, time travel, adaptive hints, built-in hints, multiple endings, science fiction, trivially-challenging, z-code, female protagonist, multiple protagonists, strong profanity, recommended for beginners, fantasy, violence, afterlife, religious, moral choice, fate', 'Title: All Roads. Author: Jon Ingold. Systems: inform. Tags: historical, time travel, sexual content, parser, alternate history, italy, science fiction, venice, electronic literature collection volume 1, hanging, assassination, male protagonist. Description: "Wave a circle round him thrice,\r\nAnd close your eyes with holy dread\r\nFor he on honey-dew hath fed\r\nAnd drunk the milk of paradise." \r\n[--blurb from Competition Aught-One]'],
    ['Systems: inform, tads, adrift. Tags: parser, fantasy, female protagonist, on jay is games, male protagonist, built-in hints, score, humor, z-code, magic, glulx, science fiction, strong npcs, systematic puzzles, single room, tads, multiple endings, second person, short, small-sized', 'Title: The Dreamhold. Author: Andrew Plotkin. Systems: inform. Tags: fantasy, recommended for beginners, amnesia, tutorial mode, adjustable difficulty, long-form, on jay is games, commercial, gender-neutral protagonist, parser, braille note apex, z-code, score, masks. Description: <i>The Dreamhold</i> is interactive fiction — a classic text adventure. No graphics! No point-and-click! You type your commands, and read what happens next.\r\n\r\n<i>The Dreamhold</i> is designed for people who have never played IF before. It introduces the common commands and mindset of text adventures, one step at a time. There’s an extensive help system describing standard IF commands, as well as dynamic hints which pop up whenever you seem to be stuck.\r\n\r\nI’ve tried to create a game which rewar'],
    ['Systems: inform, zil. Tags: parser, fantasy, commercial, infocom, score, z-code, zorkian, zork, male protagonist, collaboration, built-in hints, advanced difficulty, grue, long-form, gender-neutral protagonist, multiple endings, part of a series, full-sized, magic, maze', "Title: Vespers. Author: Jason Devlin. Systems: inform. Tags: historical, religious, adaptive hints, built-in hints, bible, monastery, monk, moral choice, christianity, middle ages, parser, plague, horror, religion, z-code, graveyard, multiple endings, male protagonist, narrative-based time. Description: It has been five days, now. Five days since I made the choice. Five days since I closed the gate.\r\n\r\nReally, there was no choice. Rovato was damned when the first spot appeared: when the first bloody cough ensued from the mouth of an urchin. To have allowed the sick sanctuary at Saint Cuthbert's would only have damned us as well.\r\n\r\nBut we were already damned.\r\n\r\nThe plague came. And now we suffer."],
    ['Systems: inform, tads, dialog. Tags: parser, female protagonist, male protagonist, built-in hints, humor, fantasy, glulx, score, mystery, single room, adaptive hints, second person, surreal, comedy, multiple endings, crime, gender-neutral protagonist, silly, horror, house setting', 'Title: Ferryman\'s Gate. Author: Daniel Maycock. Systems: inform. Tags: experimental, educational, parser, inheritance, wacky uncle. Description: You punctuation-obsessed uncle has died, leaving your family his house, but leaving you, a mere kid, with his unfinished business. As you follow the clues left by your uncle, you quickly discover that while there\'s only one way to finish the job, there are several ways to die.  An atmospheric parser game with puzzles and portals to hidden worlds. (It\'s also a thinly-veiled attempt to teach comma rules without feeling "educational.")'],
]
scores = model.predict(pairs)
print(scores)
# [-0.3367  0.9154 -0.45    0.4706 -2.2071]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Systems: twine, inform, choicescript. Tags: parser, fantasy, female protagonist, horror, science fiction, multiple endings, choice-based, short, male protagonist, humor, graphics, gender-neutral protagonist, surreal, slice of life, mystery, second person, score, choice of games, sound, built-in hints',
    [
        "Title: The Box. Author: Paul Michael Winters. Systems: kreate. Tags: escape room, parser, puzzles, second person, present tense, escape, collector, puzzle box. Description: You are a collector of rare antiquities, with a special passion for puzzle boxes. You've been searching for one particular box for a long time, but perhaps this box is better left unfound.",
        'Title: All Roads. Author: Jon Ingold. Systems: inform. Tags: historical, time travel, sexual content, parser, alternate history, italy, science fiction, venice, electronic literature collection volume 1, hanging, assassination, male protagonist. Description: "Wave a circle round him thrice,\r\nAnd close your eyes with holy dread\r\nFor he on honey-dew hath fed\r\nAnd drunk the milk of paradise." \r\n[--blurb from Competition Aught-One]',
        'Title: The Dreamhold. Author: Andrew Plotkin. Systems: inform. Tags: fantasy, recommended for beginners, amnesia, tutorial mode, adjustable difficulty, long-form, on jay is games, commercial, gender-neutral protagonist, parser, braille note apex, z-code, score, masks. Description: <i>The Dreamhold</i> is interactive fiction — a classic text adventure. No graphics! No point-and-click! You type your commands, and read what happens next.\r\n\r\n<i>The Dreamhold</i> is designed for people who have never played IF before. It introduces the common commands and mindset of text adventures, one step at a time. There’s an extensive help system describing standard IF commands, as well as dynamic hints which pop up whenever you seem to be stuck.\r\n\r\nI’ve tried to create a game which rewar',
        "Title: Vespers. Author: Jason Devlin. Systems: inform. Tags: historical, religious, adaptive hints, built-in hints, bible, monastery, monk, moral choice, christianity, middle ages, parser, plague, horror, religion, z-code, graveyard, multiple endings, male protagonist, narrative-based time. Description: It has been five days, now. Five days since I made the choice. Five days since I closed the gate.\r\n\r\nReally, there was no choice. Rovato was damned when the first spot appeared: when the first bloody cough ensued from the mouth of an urchin. To have allowed the sick sanctuary at Saint Cuthbert's would only have damned us as well.\r\n\r\nBut we were already damned.\r\n\r\nThe plague came. And now we suffer.",
        'Title: Ferryman\'s Gate. Author: Daniel Maycock. Systems: inform. Tags: experimental, educational, parser, inheritance, wacky uncle. Description: You punctuation-obsessed uncle has died, leaving your family his house, but leaving you, a mere kid, with his unfinished business. As you follow the clues left by your uncle, you quickly discover that while there\'s only one way to finish the job, there are several ways to die.  An atmospheric parser game with puzzles and portals to hidden worlds. (It\'s also a thinly-veiled attempt to teach comma rules without feeling "educational.")',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 41,261 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                               | float                                                          |
  | details | <ul><li>min: 6 tokens</li><li>mean: 72.72 tokens</li><li>max: 88 tokens</li></ul> | <ul><li>min: 30 tokens</li><li>mean: 125.88 tokens</li><li>max: 234 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.42</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                 | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Systems: twine, inform, choicescript. Tags: parser, fantasy, female protagonist, horror, science fiction, multiple endings, choice-based, short, male protagonist, humor, graphics, gender-neutral protagonist, surreal, slice of life, mystery, second person, score, choice of games, sound, built-in hints</code>                                 | <code>Title: The Box. Author: Paul Michael Winters. Systems: kreate. Tags: escape room, parser, puzzles, second person, present tense, escape, collector, puzzle box. Description: You are a collector of rare antiquities, with a special passion for puzzle boxes. You've been searching for one particular box for a long time, but perhaps this box is better left unfound.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <code>1.0</code> |
  | <code>Systems: inform, tads, zil. Tags: parser, male protagonist, slice of life, time travel, adaptive hints, built-in hints, multiple endings, science fiction, trivially-challenging, z-code, female protagonist, multiple protagonists, strong profanity, recommended for beginners, fantasy, violence, afterlife, religious, moral choice, fate</code> | <code>Title: All Roads. Author: Jon Ingold. Systems: inform. Tags: historical, time travel, sexual content, parser, alternate history, italy, science fiction, venice, electronic literature collection volume 1, hanging, assassination, male protagonist. Description: "Wave a circle round him thrice,  <br>And close your eyes with holy dread  <br>For he on honey-dew hath fed  <br>And drunk the milk of paradise."   <br>[--blurb from Competition Aught-One]</code>                                                                                                                                                                                                                                                                                                                                                                | <code>1.0</code> |
  | <code>Systems: inform, tads, adrift. Tags: parser, fantasy, female protagonist, on jay is games, male protagonist, built-in hints, score, humor, z-code, magic, glulx, science fiction, strong npcs, systematic puzzles, single room, tads, multiple endings, second person, short, small-sized</code>                                                     | <code>Title: The Dreamhold. Author: Andrew Plotkin. Systems: inform. Tags: fantasy, recommended for beginners, amnesia, tutorial mode, adjustable difficulty, long-form, on jay is games, commercial, gender-neutral protagonist, parser, braille note apex, z-code, score, masks. Description: <i>The Dreamhold</i> is interactive fiction — a classic text adventure. No graphics! No point-and-click! You type your commands, and read what happens next.  <br>  <br><i>The Dreamhold</i> is designed for people who have never played IF before. It introduces the common commands and mindset of text adventures, one step at a time. There’s an extensive help system describing standard IF commands, as well as dynamic hints which pop up whenever you seem to be stuck.  <br>  <br>I’ve tried to create a game which rewar</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.1939 | 500  | 0.8540        |
| 0.3877 | 1000 | 0.6777        |
| 0.5816 | 1500 | 0.6201        |
| 0.7755 | 2000 | 0.6168        |
| 0.9694 | 2500 | 0.6097        |
| 1.1632 | 3000 | 0.5997        |
| 1.3571 | 3500 | 0.5881        |
| 1.5510 | 4000 | 0.5797        |
| 1.7449 | 4500 | 0.5822        |
| 1.9387 | 5000 | 0.5684        |


### Training Time
- **Training**: 23.6 minutes

### Framework Versions
- Python: 3.14.4
- Sentence Transformers: 5.4.1
- Transformers: 5.6.2
- PyTorch: 2.11.0
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->