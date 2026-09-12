---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:39276
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
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision 233902d25c440f23af6f7d6e94d2946bac0bee0a -->
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
    ['Systems: inform, custom, twine. Tags: puzzle, gender-neutral protagonist, limited verbs, graded success, parser, score, glulx, multiple endings, male protagonist, wordplay, constrained protagonist, surreal, fantasy, magic, test as premise, systematic puzzles, spells, mathematics, logic puzzles, academia. Authors: Arthur DiBianca, Mike Spivey, Nicky Case, William Rous, Emily Short', 'Title: Junior Arithmancer. Author: Mike Spivey. Systems: inform. Tags: surreal, fantasy, parser, gender-neutral protagonist, score, magic, glulx, test as premise, systematic puzzles, spells, graded success, mathematics, logic puzzles, academia, unmnemonic. Language: english. Description: A one-to-many-room puzzler.'],
    ['Systems: twine, inform, custom. Tags: parser, female protagonist, horror, choice-based, multiple endings, fantasy, graphics, music, science fiction, mystery, glulx, magic, cyoa with location-based world model, male protagonist, surreal, sound, sugarcube, humor, built-in hints, lgbtq+. Authors: Ryan Veeder, Jamwitch, Coral Nulla, StamblerRambler, Ben Jackson', 'Title: The Little Match Girl 3: The Escalus Manifold. Author: Ryan Veeder. Systems: inform. Tags: adventure, rpg, parser, female protagonist, violence, sound, music, magic, combat, third person, time travel, fairy tale, witch, vorple, random combat, witches, distinct areas, part of a series, gun, spells. Language: english. Description: "The Snow Queen controls her servants with Shards from the Mirror of Belial," Ebenezer Scrooge explained.'],
    ['Systems: inform, chooseyourstory, twine. Tags: parser, choice-based, fantasy, male protagonist, science fiction, female protagonist, chooseyourstory, score, violence, multiple endings, gender-neutral protagonist, humor, second person, built-in hints, magic, horror, historical, nonhuman protagonist, glulx, mystery. Authors: Ryan Veeder, Will11, Arthur DiBianca, Sindriv, C.E.J. Pacian. Dislikes: speedif, short, single room, surreal, tads 2, joke, satire, they might be giants, word count limit, speed if', 'Title: The Paper Magician. Author: Soojung Choi. Systems: twine. Tags: speculative fiction, fantasy, choice-based, gender-neutral protagonist, first effort, first person, cyoa with location-based world model, escape, cats, science fantasy. Language: english. Description: "The cat leapt for the star." Every time I conjured a cat and a star from the words I penned on paper, I found myself more and more like the cat. Longing for something bright and hopeful, but forever out of my reach.\r\n<br/>\r\nUntil one night when I dreamed for the first time... I met someone who promised to help me escape.\r\n<br/>\r\nThis is the story of a test subject, their encounter with a spirit, and their escape from a mysterious lab.'],
    ['Systems: inform, twine, tads. Tags: parser, female protagonist, fantasy, male protagonist, humor, multiple endings, horror, gender-neutral protagonist, mystery, built-in hints, surreal, score, graphics, second person, choice-based, science fiction, glulx, violence, slice of life, present tense. Authors: Ryan Veeder, Emily Short, Eric Eve, Andrew Schultz, Chandler Groover. Dislikes: cyoa, short, bookmarks, profanity, landscape mode, third person, sound, speedif, hypertext, romance', 'Title: Bolivia By Night. Author: Aidan Doyle. Systems: tads. Tags: mystery, fantasy, travel, parser, female protagonist, graphics, built-in hints, tads, tads 2, gender choice, chapters, bolivia. Language: english. Description: A mystery adventure game set in Bolivia.'],
    ["Systems: inform, tads, adrift. Tags: parser, male protagonist, on jay is games, female protagonist, z-code, glulx, built-in hints, small-sized, magic, score, fantasy, science fiction, humor, single room, multiple endings, tads, historical, time travel, short, systematic puzzles. Authors: Eric Eve, Adam Cadre (as Opal O'Donnell), Jon Ingold, Jeremy Freese, G. Kevin Wilson. Dislikes: first effort, horror, animal protagonist, mystery, surreal, nonhuman protagonist, present tense, amnesia, apartment, cave", "Title: Foo Foo. Author: Buster Hudson. Systems: inform. Tags: mystery, parser, female protagonist, multiple endings, nonhuman protagonist, second person, lgbtq+, present tense, gay/queer protagonist, glulx, crime, hints, back garden, noir, npc based hint system, anthropomorphised animals, rat, rabbit, fairies, rvexpo. Language: english. Description: Someone's been bopping the field mice on the head, and only Good Fairy, Senior Detective can find out who.\r\n\r\nA parser-driven noir adventure based on the interactive fiction of Ryan Veeder."],
]
scores = model.predict(pairs)
print(scores)
# [ 6.871   5.8048 -0.0761 -0.7185 -2.2642]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Systems: inform, custom, twine. Tags: puzzle, gender-neutral protagonist, limited verbs, graded success, parser, score, glulx, multiple endings, male protagonist, wordplay, constrained protagonist, surreal, fantasy, magic, test as premise, systematic puzzles, spells, mathematics, logic puzzles, academia. Authors: Arthur DiBianca, Mike Spivey, Nicky Case, William Rous, Emily Short',
    [
        'Title: Junior Arithmancer. Author: Mike Spivey. Systems: inform. Tags: surreal, fantasy, parser, gender-neutral protagonist, score, magic, glulx, test as premise, systematic puzzles, spells, graded success, mathematics, logic puzzles, academia, unmnemonic. Language: english. Description: A one-to-many-room puzzler.',
        'Title: The Little Match Girl 3: The Escalus Manifold. Author: Ryan Veeder. Systems: inform. Tags: adventure, rpg, parser, female protagonist, violence, sound, music, magic, combat, third person, time travel, fairy tale, witch, vorple, random combat, witches, distinct areas, part of a series, gun, spells. Language: english. Description: "The Snow Queen controls her servants with Shards from the Mirror of Belial," Ebenezer Scrooge explained.',
        'Title: The Paper Magician. Author: Soojung Choi. Systems: twine. Tags: speculative fiction, fantasy, choice-based, gender-neutral protagonist, first effort, first person, cyoa with location-based world model, escape, cats, science fantasy. Language: english. Description: "The cat leapt for the star." Every time I conjured a cat and a star from the words I penned on paper, I found myself more and more like the cat. Longing for something bright and hopeful, but forever out of my reach.\r\n<br/>\r\nUntil one night when I dreamed for the first time... I met someone who promised to help me escape.\r\n<br/>\r\nThis is the story of a test subject, their encounter with a spirit, and their escape from a mysterious lab.',
        'Title: Bolivia By Night. Author: Aidan Doyle. Systems: tads. Tags: mystery, fantasy, travel, parser, female protagonist, graphics, built-in hints, tads, tads 2, gender choice, chapters, bolivia. Language: english. Description: A mystery adventure game set in Bolivia.',
        "Title: Foo Foo. Author: Buster Hudson. Systems: inform. Tags: mystery, parser, female protagonist, multiple endings, nonhuman protagonist, second person, lgbtq+, present tense, gay/queer protagonist, glulx, crime, hints, back garden, noir, npc based hint system, anthropomorphised animals, rat, rabbit, fairies, rvexpo. Language: english. Description: Someone's been bopping the field mice on the head, and only Good Fairy, Senior Detective can find out who.\r\n\r\nA parser-driven noir adventure based on the interactive fiction of Ryan Veeder.",
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

* Size: 39,276 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                           | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                               | float                                                          |
  | details | <ul><li>min: 12 tokens</li><li>mean: 121.1 tokens</li><li>max: 150 tokens</li></ul> | <ul><li>min: 40 tokens</li><li>mean: 129.72 tokens</li><li>max: 245 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.47</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Systems: inform, custom, twine. Tags: puzzle, gender-neutral protagonist, limited verbs, graded success, parser, score, glulx, multiple endings, male protagonist, wordplay, constrained protagonist, surreal, fantasy, magic, test as premise, systematic puzzles, spells, mathematics, logic puzzles, academia. Authors: Arthur DiBianca, Mike Spivey, Nicky Case, William Rous, Emily Short</code>                                                                                                                            | <code>Title: Junior Arithmancer. Author: Mike Spivey. Systems: inform. Tags: surreal, fantasy, parser, gender-neutral protagonist, score, magic, glulx, test as premise, systematic puzzles, spells, graded success, mathematics, logic puzzles, academia, unmnemonic. Language: english. Description: A one-to-many-room puzzler.</code>                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>1.0</code> |
  | <code>Systems: twine, inform, custom. Tags: parser, female protagonist, horror, choice-based, multiple endings, fantasy, graphics, music, science fiction, mystery, glulx, magic, cyoa with location-based world model, male protagonist, surreal, sound, sugarcube, humor, built-in hints, lgbtq+. Authors: Ryan Veeder, Jamwitch, Coral Nulla, StamblerRambler, Ben Jackson</code>                                                                                                                                                   | <code>Title: The Little Match Girl 3: The Escalus Manifold. Author: Ryan Veeder. Systems: inform. Tags: adventure, rpg, parser, female protagonist, violence, sound, music, magic, combat, third person, time travel, fairy tale, witch, vorple, random combat, witches, distinct areas, part of a series, gun, spells. Language: english. Description: "The Snow Queen controls her servants with Shards from the Mirror of Belial," Ebenezer Scrooge explained.</code>                                                                                                                                                                                                                                                                                         | <code>1.0</code> |
  | <code>Systems: inform, chooseyourstory, twine. Tags: parser, choice-based, fantasy, male protagonist, science fiction, female protagonist, chooseyourstory, score, violence, multiple endings, gender-neutral protagonist, humor, second person, built-in hints, magic, horror, historical, nonhuman protagonist, glulx, mystery. Authors: Ryan Veeder, Will11, Arthur DiBianca, Sindriv, C.E.J. Pacian. Dislikes: speedif, short, single room, surreal, tads 2, joke, satire, they might be giants, word count limit, speed if</code> | <code>Title: The Paper Magician. Author: Soojung Choi. Systems: twine. Tags: speculative fiction, fantasy, choice-based, gender-neutral protagonist, first effort, first person, cyoa with location-based world model, escape, cats, science fantasy. Language: english. Description: "The cat leapt for the star." Every time I conjured a cat and a star from the words I penned on paper, I found myself more and more like the cat. Longing for something bright and hopeful, but forever out of my reach.  <br><br/>  <br>Until one night when I dreamed for the first time... I met someone who promised to help me escape.  <br><br/>  <br>This is the story of a test subject, their encounter with a spirit, and their escape from a mysterious lab.</code> | <code>1.0</code> |
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
| 0.2037 | 500  | 0.8332        |
| 0.4073 | 1000 | 0.6618        |
| 0.6110 | 1500 | 0.6080        |
| 0.8147 | 2000 | 0.5850        |
| 1.0183 | 2500 | 0.5575        |
| 1.2220 | 3000 | 0.5449        |
| 1.4257 | 3500 | 0.5357        |
| 1.6293 | 4000 | 0.5274        |
| 1.8330 | 4500 | 0.5118        |


### Training Time
- **Training**: 29.2 minutes

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