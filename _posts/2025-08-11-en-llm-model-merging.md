---
layout: article
lang: en
title: "A Practical Guide to Merging LLMs: Compatibility, Methods, and Evaluation"
description: "Understand when model merging is plausible, how task-vector methods differ, what compatibility checks matter, and how to evaluate a merged checkpoint safely."
date: 2025-08-11 13:33:51 +0000
permalink: /writing/llm-model-merging/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# A Practical Guide to Merging LLMs: Compatibility, Methods, and Evaluation

Model merging combines parameters from trained models into one checkpoint. It can be a useful way to explore whether related fine-tunes can share capabilities, but it is not a general-purpose way to combine arbitrary models. The weights must have a meaningful correspondence, and the merged result needs to be evaluated like any other model artifact.

Merging also does not automatically make inference faster or cheaper. A merged dense model usually has the same architecture and similar parameter count as its inputs. Combining weights is different from serving an ensemble, distilling a smaller model, or changing the architecture.

## What is being merged?

Let `theta_0` be a base model's parameters and `theta_A` and `theta_B` be fine-tuned versions of that same base. Their task vectors are the parameter changes introduced by fine-tuning:

```text
task_vector_A = theta_A - theta_0
task_vector_B = theta_B - theta_0
```

A weighted task-vector merge can be written as:

```text
theta_merged = theta_0 + weight_A * task_vector_A + weight_B * task_vector_B
```

The coefficients control how much of each change is added. This compact equation hides important assumptions: corresponding tensors need compatible shapes and meaning; the base needs to be the correct ancestor; and useful changes from one fine-tune must not damage another capability when combined.

The [Task Arithmetic paper](https://arxiv.org/abs/2212.04089) defines task vectors as differences between a pretrained model and a model fine-tuned from it. Later methods such as [TIES-Merging](https://arxiv.org/abs/2306.01708) try to reduce conflicts among those parameter changes.

## Check compatibility before choosing a method

Before loading large checkpoints, compare their model configurations, parameter names, tensor shapes, tokenizer files, vocabulary, special tokens, and training lineage. A merge tool can often detect incompatible tensor shapes; passing that check does not prove that weights are semantically aligned.

For task-vector arithmetic, use fine-tunes with a known shared base whenever possible. If the base differs, the parameter deltas may reflect different training histories rather than task-specific changes that can be safely added. Model name similarity is not evidence of a shared base.

Pay special attention to tokenizer and embedding changes. If a fine-tune adds tokens or changes vocabulary rows, the corresponding embedding and output tensors need an explicit, validated mapping. Copying or averaging rows by position can silently associate the wrong token with a vector.

Finally, review model licenses and redistribution terms. A technical merge does not erase the obligations attached to the input checkpoints. Keep the exact model revisions, sources, licenses, and merge configuration with the output artifact.

## A small arithmetic example

This pure-Python example shows task-vector arithmetic with two-coordinate vectors. It is a toy calculation, not a neural-network merge and not a performance result.

```python
base = [1.0, 0.0]
fine_tune_a = [1.4, 0.2]
fine_tune_b = [0.8, 0.5]

weight_a = 0.5
weight_b = 0.5

assert len(base) == len(fine_tune_a) == len(fine_tune_b)

delta_a = [a - b for a, b in zip(fine_tune_a, base)]
delta_b = [a - b for a, b in zip(fine_tune_b, base)]
merged = [
    b + weight_a * da + weight_b * db
    for b, da, db in zip(base, delta_a, delta_b)
]

print(merged)  # [1.1, 0.35]
```

In a real checkpoint, every parameter tensor has many values, and the task-vector changes can conflict. The arithmetic is easy; determining whether the result retains the desired behavior is the hard part.

## Common merging approaches

| Method | Core idea | Useful when | Main caution |
| --- | --- | --- | --- |
| Model soups / linear averaging | Average compatible checkpoints or fine-tunes | Checkpoints are closely related and lie in a compatible region of parameter space | Averages can hurt if the inputs are not aligned or differ substantially |
| SLERP | Interpolate between two parameter vectors on a spherical path | Two compatible checkpoints provide a useful interpolation range | Matching shapes alone does not guarantee meaningful interpolation |
| Task arithmetic | Add weighted fine-tuning deltas to a shared base | Several fine-tunes share a known ancestor | Task changes can interfere or amplify unwanted behavior |
| TIES-Merging | Trim small changes, resolve sign conflicts, then merge aligned changes | Task-vector interference is a concern | The method still depends on compatible task vectors and evaluation |
| DARE | Randomly drop parts of task vectors and rescale retained changes; variants can combine this with TIES | Sparse task-vector experiments are justified | Results depend on model, task, and sampling; it is not a universal improvement |
| Passthrough / layer composition | Copy selected layers or tensors into a composed checkpoint | You intentionally build a layer stack from compatible donor layers | This is layer composition, not a general merge across architectures |

The [MergeKit method guide](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md) documents current implementations and their input requirements. Its `passthrough` method copies tensors from a selected input and is used for layer-stacking configurations; it does not make unrelated architectures interchangeable.

The [Model Soups paper](https://arxiv.org/abs/2203.05482) studies averaging multiple fine-tuned models in settings where their weights are sufficiently compatible. The [TIES paper](https://arxiv.org/abs/2306.01708) describes three steps: trimming small changes, resolving sign disagreements, and merging parameter changes that align with the selected sign. [DARE](https://arxiv.org/abs/2311.03099) explores dropping and rescaling parts of task vectors. These methods offer hypotheses to test, not guarantees that a particular LLM merge will work.

## Evaluate the output, not just whether the merge completed

A completed command only means that the tool wrote a checkpoint. Build an evaluation plan before tuning coefficients or selecting layers:

1. **Keep the inputs as baselines.** Evaluate the base and every source fine-tune on the same task set and runtime.
2. **Choose held-out tasks.** Include target capabilities, general behavior, and tasks where regression would be costly. Keep tuning data separate from final evaluation data.
3. **Check the interface.** Verify that tokenizer, chat template, generation configuration, and any custom model code match the merged checkpoint.
4. **Measure both gains and regressions.** Compare task quality, refusal and safety behavior, instruction following, multilingual behavior where relevant, and output consistency.
5. **Repeat stochastic evaluations.** Record sampling settings and use enough trials to understand variation. Do not select a merge by one favorable example.
6. **Inspect the artifact.** Reload it from a clean environment, run a short inference smoke test, and retain its provenance and license information.

If a coefficient is selected by repeatedly checking the final test set, that set has become tuning data. Reserve an untouched holdout for the final comparison. Record model revisions, tool version, configuration, hardware, evaluation date, and raw results so someone can reproduce the decision.

## Common failure modes

### “They have the same size, so I can merge them”

Equal tensor dimensions do not establish that parameter positions have compatible meanings. Check architecture, tensor naming, tokenizer, training lineage, and the exact merge method's input contract.

### “The models share a family name, so their deltas are compatible”

Families can include different base revisions, vocabulary changes, architecture variants, or training recipes. Confirm the exact source checkpoint and configuration rather than relying on labels.

### “A merged model gets both capabilities for free”

Capabilities can interfere. A merge may improve one task and regress another, or change behavior in ways not captured by a narrow benchmark. Evaluate against both intended uses and important non-target behavior.

### “Passthrough merges any models”

Passthrough copies tensors or layer ranges. The selected layers still need to fit the composed architecture and provide valid interfaces to their neighboring layers. It is a construction tool, not a universal compatibility adapter.

### “Merging reduces serving cost”

Weight merging does not inherently reduce parameter count, memory requirements, or latency. A deployment change needs its own measurement. To reduce inference cost, investigate quantization, distillation, sparse architectures, or smaller models separately and evaluate their trade-offs.

## When to use a different technique

Use continued fine-tuning when you have data and compute to train the desired behavior directly. Use an ensemble when independent outputs are valuable and additional serving cost is acceptable. Use distillation when the goal is to move behavior into a smaller student model. Use retrieval when the missing capability is access to changing or private information. Each option has different data, compute, latency, licensing, and evaluation requirements.

Model merging is most useful as a controlled experiment with compatible checkpoints, a clear target, and a regression suite. Start with a shared base and simple arithmetic, keep the input artifacts immutable, and retain the merge configuration. Adopt the result only when the measured gains hold on your own workload and the regressions are acceptable.

For related approaches, see [fine-tuning LLMs (in Persian)](/fa/writing/efficient-fine-tuning/) and [knowledge distillation (in Persian)](/fa/writing/knowledge-distillation/).

## References

- [Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time](https://arxiv.org/abs/2203.05482)
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)
- [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)
- [MergeKit: Merge Method Guide](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md)
