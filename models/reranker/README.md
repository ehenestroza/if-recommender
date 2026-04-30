---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:40577
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
    ['Systems: inform, twine, storynexus. Tags: parser, gender-neutral protagonist, fantasy, choice-based, horror, on jay is games, recommended for beginners, second person, limited verbs, multiple endings, cyoa, graphics, puzzleless, short, surreal, linear, present tense, humor, memorable npc, commercial', 'Title: Open That Vein. Author: Chandler Groover. Systems: inform. Tags: horror, parser, linear, limited verbs, gender-neutral protagonist, dynamic fiction, body horror, blood, present tense, second person, speed if, surreal, wolf, la petite mort. Description: You are going to open that vein.\r\n\r\nLa Petite Mort entry in ECTOCOMP 2015.'],
    ['Systems: inform, tads, hugo. Tags: parser, science fiction, horror, male protagonist, violence, surreal, built-in hints, female protagonist, sexual content, slice of life, on jay is games, humor, score, strong profanity, first person, graphics, amnesia, nonhuman protagonist, fantasy, multiple endings', 'Title: Punk Points. Author: Jim Munroe. Systems: inform. Tags: slice of life, mild profanity, sexual content, punk, coming of age, male protagonist, teenage protagonist, smoking, high school. Description: "It\'s the first day of high school and you\'ve decided to give yourself a mohawk. Now you\'ve gotta stand up to teachers, impress peers and make a name for yourself until you\'ve earned enough Punk Points to escape the suburbs." [--blurb from Competition Aught-Zero]'],
    ['Systems: inform, hugo. Tags: parser, on jay is games, small-sized, recommended for beginners, puzzleless, fantasy, humor, nonhuman protagonist, female protagonist, character based, female, plot, character, horror, multiple endings, trivially-challenging, science fiction, male protagonist, strong profanity, slice of life', 'Title: Shade. Author: Andrew Plotkin. Systems: inform. Tags: travel, puzzleless, one-room, gender-neutral protagonist, single room, surreal, desert, changing environment, house setting, atmospheric, on jay is games, commercial, sand, parser, unreliable narrator, horror, recommended for beginners, let\'s play, apartment, z-code. Description: "A one-room game set in your apartment." [--blurb from Competition Aught-Zero]'],
    ['Systems: twine, inform, choicescript. Tags: parser, fantasy, horror, female protagonist, science fiction, multiple endings, choice-based, short, graphics, humor, male protagonist, gender-neutral protagonist, surreal, slice of life, mystery, second person, score, choice of games, sound, nonhuman protagonist', "Title: Focal Shift. Author: Fred Snyder. Systems: gamefic. Tags: cyberpunk, hacking, parser, gamefic. Description: All you were supposed to do is sneak into a fintech company and steal some data. You didn't expect the client to use the comm chip jacked into your cybernetic implant as a cattle prod. And you definitely didn't expect a murder to be part of the plan. Good luck hacking your way out of this jam, grid jockey."],
    ['Systems: twine, inform, ink. Tags: parser, short, horror, female protagonist, fantasy, lgbtq+, multiple endings, choice-based, gender-neutral protagonist, science fiction, surreal, slice of life, humor, word count limit, mystery, romance, graphics, male protagonist, nonhuman protagonist, music', 'Title: Mooncrash!. Author: Laura. Systems: inform. Tags: fantasy, apocalyptic, rpg elements, philosophical, parser, science fantasy, sword, tower, philosophical fantasy, multiple endings, opening questionnaire, maze, gun, first effort, dragons, fast-paced, dragon, dagger, demon, conversation puzzles. Description: You are one of the pre-eminent heroes of this world, working directly under The Tempest Council. You fight to prevent the end of the world by any means necessary. You are either the best at what you do, or only an actual member of The Council has a credible claim to be better. The world is your oyster.\r\n<br/>\r\nOr, at least it was, until yesterday.\r\n<br/>\r\nFor centuries, The Tempest Council has watched helplessly as the threads of fate coalesced into a portent of destruction. Even with all their'],
]
scores = model.predict(pairs)
print(scores)
# [ 2.5536  0.5405  0.3738 -0.1681  0.1808]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Systems: inform, twine, storynexus. Tags: parser, gender-neutral protagonist, fantasy, choice-based, horror, on jay is games, recommended for beginners, second person, limited verbs, multiple endings, cyoa, graphics, puzzleless, short, surreal, linear, present tense, humor, memorable npc, commercial',
    [
        'Title: Open That Vein. Author: Chandler Groover. Systems: inform. Tags: horror, parser, linear, limited verbs, gender-neutral protagonist, dynamic fiction, body horror, blood, present tense, second person, speed if, surreal, wolf, la petite mort. Description: You are going to open that vein.\r\n\r\nLa Petite Mort entry in ECTOCOMP 2015.',
        'Title: Punk Points. Author: Jim Munroe. Systems: inform. Tags: slice of life, mild profanity, sexual content, punk, coming of age, male protagonist, teenage protagonist, smoking, high school. Description: "It\'s the first day of high school and you\'ve decided to give yourself a mohawk. Now you\'ve gotta stand up to teachers, impress peers and make a name for yourself until you\'ve earned enough Punk Points to escape the suburbs." [--blurb from Competition Aught-Zero]',
        'Title: Shade. Author: Andrew Plotkin. Systems: inform. Tags: travel, puzzleless, one-room, gender-neutral protagonist, single room, surreal, desert, changing environment, house setting, atmospheric, on jay is games, commercial, sand, parser, unreliable narrator, horror, recommended for beginners, let\'s play, apartment, z-code. Description: "A one-room game set in your apartment." [--blurb from Competition Aught-Zero]',
        "Title: Focal Shift. Author: Fred Snyder. Systems: gamefic. Tags: cyberpunk, hacking, parser, gamefic. Description: All you were supposed to do is sneak into a fintech company and steal some data. You didn't expect the client to use the comm chip jacked into your cybernetic implant as a cattle prod. And you definitely didn't expect a murder to be part of the plan. Good luck hacking your way out of this jam, grid jockey.",
        'Title: Mooncrash!. Author: Laura. Systems: inform. Tags: fantasy, apocalyptic, rpg elements, philosophical, parser, science fantasy, sword, tower, philosophical fantasy, multiple endings, opening questionnaire, maze, gun, first effort, dragons, fast-paced, dragon, dagger, demon, conversation puzzles. Description: You are one of the pre-eminent heroes of this world, working directly under The Tempest Council. You fight to prevent the end of the world by any means necessary. You are either the best at what you do, or only an actual member of The Council has a credible claim to be better. The world is your oyster.\r\n<br/>\r\nOr, at least it was, until yesterday.\r\n<br/>\r\nFor centuries, The Tempest Council has watched helplessly as the threads of fate coalesced into a portent of destruction. Even with all their',
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

* Size: 40,577 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                               | float                                                          |
  | details | <ul><li>min: 6 tokens</li><li>mean: 72.88 tokens</li><li>max: 85 tokens</li></ul> | <ul><li>min: 30 tokens</li><li>mean: 125.08 tokens</li><li>max: 266 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.46</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Systems: inform, twine, storynexus. Tags: parser, gender-neutral protagonist, fantasy, choice-based, horror, on jay is games, recommended for beginners, second person, limited verbs, multiple endings, cyoa, graphics, puzzleless, short, surreal, linear, present tense, humor, memorable npc, commercial</code>                      | <code>Title: Open That Vein. Author: Chandler Groover. Systems: inform. Tags: horror, parser, linear, limited verbs, gender-neutral protagonist, dynamic fiction, body horror, blood, present tense, second person, speed if, surreal, wolf, la petite mort. Description: You are going to open that vein.  <br>  <br>La Petite Mort entry in ECTOCOMP 2015.</code>                                                                                                                                 | <code>1.0</code> |
  | <code>Systems: inform, tads, hugo. Tags: parser, science fiction, horror, male protagonist, violence, surreal, built-in hints, female protagonist, sexual content, slice of life, on jay is games, humor, score, strong profanity, first person, graphics, amnesia, nonhuman protagonist, fantasy, multiple endings</code>                     | <code>Title: Punk Points. Author: Jim Munroe. Systems: inform. Tags: slice of life, mild profanity, sexual content, punk, coming of age, male protagonist, teenage protagonist, smoking, high school. Description: "It's the first day of high school and you've decided to give yourself a mohawk. Now you've gotta stand up to teachers, impress peers and make a name for yourself until you've earned enough Punk Points to escape the suburbs." [--blurb from Competition Aught-Zero]</code> | <code>1.0</code> |
  | <code>Systems: inform, hugo. Tags: parser, on jay is games, small-sized, recommended for beginners, puzzleless, fantasy, humor, nonhuman protagonist, female protagonist, character based, female, plot, character, horror, multiple endings, trivially-challenging, science fiction, male protagonist, strong profanity, slice of life</code> | <code>Title: Shade. Author: Andrew Plotkin. Systems: inform. Tags: travel, puzzleless, one-room, gender-neutral protagonist, single room, surreal, desert, changing environment, house setting, atmospheric, on jay is games, commercial, sand, parser, unreliable narrator, horror, recommended for beginners, let's play, apartment, z-code. Description: "A one-room game set in your apartment." [--blurb from Competition Aught-Zero]</code>                                                 | <code>0.0</code> |
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
| 0.1971 | 500  | 0.8616        |
| 0.3942 | 1000 | 0.6907        |
| 0.5912 | 1500 | 0.6235        |
| 0.7883 | 2000 | 0.6157        |
| 0.9854 | 2500 | 0.5986        |
| 1.1825 | 3000 | 0.6005        |
| 1.3796 | 3500 | 0.5942        |
| 1.5767 | 4000 | 0.5796        |
| 1.7737 | 4500 | 0.5838        |
| 1.9708 | 5000 | 0.5665        |


### Training Time
- **Training**: 23.4 minutes

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