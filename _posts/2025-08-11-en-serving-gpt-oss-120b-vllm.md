---
layout: article
lang: en
title: "Serving GPT-OSS 120B with vLLM: Capacity, Configuration, and Security"
description: "Plan a GPT-OSS 120B deployment by separating weight fit from runtime capacity, testing the exact vLLM stack, and protecting every serving endpoint."
date: 2025-08-11 03:24:59 +0000
permalink: /writing/serving-gpt-oss-120b-vllm/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Serving GPT-OSS 120B with vLLM: Capacity, Configuration, and Security

GPT-OSS 120B is an open-weight mixture-of-experts model that can be served with vLLM on supported hardware. Deployment planning should start with the actual checkpoint, context length, concurrency, and network boundary. A model that loads successfully at low concurrency is not automatically ready for production traffic.

The guidance below is for the `openai/gpt-oss-120b` checkpoint family. Upstream recipes and supported accelerators change over time. Pin a known model revision, vLLM release, container or environment, and hardware configuration before treating a command as reproducible.

## Understand the memory requirement

OpenAI's [GPT-OSS architecture and model card](https://deploymentsafety.openai.com/gpt-oss/architecture) reports 116.8 billion total parameters, about 5.1 billion active parameters per token on a forward pass, and a 60.8 GiB checkpoint size for the 120B model. The release uses MXFP4 quantization for most MoE weights. Active parameters describe the subset used for a token's computation; they do not describe the amount of weight data that must be stored for inference.

The [vLLM GPT-OSS recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md) says its 120B checkpoint can fit on one A100 80GB, while tensor parallelism across more GPUs can help throughput. Treat this as a supported starting point from that recipe, not a guarantee for every context, batch size, vLLM version, or serving workload.

GPU memory is also used by the KV cache, activations, runtime workspaces, and concurrent requests. Longer contexts and larger batches can increase those needs. Before provisioning a paid machine, decide on a context target and expected concurrency, then measure peak memory with the exact launch configuration. If the model loads only after reducing the context or batch limit, record that as a capacity constraint rather than claiming the original target works.

## Start from the maintained recipe

The simplest upstream command is:

```bash
vllm serve openai/gpt-oss-120b
```

That command is a quick start, not a complete production configuration. Follow the current [vLLM GPT-OSS recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md) for the accelerator family and serving mode you choose. Pin the vLLM release and model revision in your deployment. Do not copy settings from a different GPU, quantization format, or benchmark configuration without checking what those settings change.

The recipe includes different paths for NVIDIA and AMD hardware and discusses tensor parallelism. More GPUs can improve throughput in some deployments, but add interconnect, coordination, and idle capacity costs. Measure both low-concurrency responsiveness and the batch or concurrency level that represents your service.

## Separate a quick smoke test from a capacity benchmark

After installation, a small request can confirm that the server loads the model and returns a completion. That is a smoke test. It does not establish production throughput, tail latency, tool reliability, or behavior at the intended context length.

For a capacity evaluation, use a representative prompt-length and output-length distribution. Record at least:

- Model revision, vLLM version, container digest, and launch arguments.
- GPU model and count, interconnect, host CPU and memory, and driver/runtime details.
- Input and output token distributions, request concurrency, and batch settings.
- Time to first token, inter-token latency, throughput, error rate, and peak memory.
- Whether prefix caching, speculative decoding, tool calls, or other options were enabled.

Repeat the same workload after each configuration change. Separate changes that alter model output from changes that only alter scheduling or hardware. Do not reuse third-party benchmark numbers as if they were measured in your environment.

## Secure the HTTP server before exposing it

Do not expose a development server directly to the public internet. Bind it to a private interface or keep it behind a trusted reverse proxy, firewall, or private service network. Protect the model endpoint and the operational endpoints with the same access policy.

This matters because the current [vLLM security documentation](https://docs.vllm.ai/en/latest/usage/security/) says `--api-key` only protects particular API path prefixes. Other endpoints, including `/invocations`, may expose inference without that key; additional operational endpoints can also be unprotected. An API key flag alone is not a complete perimeter. Inspect the endpoint list for the exact vLLM version, block routes that are not needed, and apply authentication and network controls at a trusted proxy.

Use scoped credentials, TLS at the service boundary, request limits, and logs that avoid storing unnecessary prompt content. If a hosted GPU provider gives the instance a public address, verify the firewall and inbound rules before starting the server. Keep administrative access separate from the inference endpoint.

## Plan the deployment around workload and budget

Estimate the deployment from measured workload rather than headline parameter counts. Include GPU time, storage for checkpoint and cache, startup and model download time, idle capacity, network transfer, and the cost of failed or retried requests. A reserved multi-GPU machine can be wasteful for a low-volume endpoint; a single GPU may be inadequate for a long-context, high-concurrency service.

Define the latency and availability target before selecting parallelism. Then test whether the same target can be met after including health checks, warm-up, queueing, and deployment restart behavior. Add admission control and request limits so bursts do not consume all cache memory or make every user wait behind an oversized request.

## A deployment checklist

Before allowing real users or sensitive data, verify:

- The exact model revision and license terms are recorded.
- The chosen vLLM release supports the model and accelerator combination.
- The longest expected context and peak concurrency fit the measured memory budget.
- The service passes a small inference smoke test and a representative load test.
- Every inference and administrative endpoint is reachable only through intended controls.
- Monitoring covers readiness, errors, queueing, GPU memory, request latency, and process restarts.
- Rollback can restore the previous model and serving configuration.

The upstream command is a starting point. A production deployment is a separately verified system consisting of a model revision, runtime, hardware, limits, security boundary, and measured workload. Record that configuration so an update to vLLM or the model does not silently change the assumptions.

For related deployment decisions, see [local LLM inference](/fa/writing/local-llm-inference/) and [how to evaluate reasoning effort](/writing/inkling-effort-evaluation/).

## References

- [OpenAI GPT-OSS architecture and model card](https://deploymentsafety.openai.com/gpt-oss/architecture)
- [vLLM GPT-OSS recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md)
- [vLLM serving security](https://docs.vllm.ai/en/latest/usage/security/)
- [vLLM serve command](https://docs.vllm.ai/en/latest/cli/serve/)
