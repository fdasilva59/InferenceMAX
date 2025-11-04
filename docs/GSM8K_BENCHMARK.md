# GSM8k Benchmark CI

This directory contains the CI workflow for running GSM8k (Grade School Math 8K) benchmarks using lm-evaluation-harness.

## Overview

GSM8k is a dataset of 8.5K high-quality linguistically diverse grade school math word problems. This benchmark evaluates the mathematical reasoning capabilities of language models.

## Workflow

The GSM8k benchmark CI is implemented in `.github/workflows/gsm8k-benchmark.yml` and can be triggered manually via workflow_dispatch.

### Parameters

- **model**: The model to evaluate (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- **runner**: The GPU runner to use (e.g., `h100-cr`, `h200-cw`, `b200-nv`, `mi300x-amd`, `mi325x-amd`, `mi355x-amd`)
- **image**: Docker image to use (e.g., `vllm/vllm-openai:latest`)
- **framework**: Inference framework (`vllm` or `sglang`)
- **precision**: Model precision (`fp16`, `fp8`, `fp4`)
- **tp**: Tensor parallelism size
- **num-fewshot**: Number of few-shot examples (default: 5)
- **limit**: Optional limit on number of examples to evaluate

## Architecture

### Benchmark Scripts

Located in `benchmarks/`:
- `gsm8k_vllm_docker.sh`: Runs GSM8k evaluation using vLLM
- `gsm8k_sglang_docker.sh`: Runs GSM8k evaluation using SGLang

These scripts:
1. Start an inference server (vLLM or SGLang)
2. Wait for the server to be ready
3. Run lm-eval with GSM8k task
4. Save results and shutdown the server

### Runner Scripts

Located in `runners/`:
- `launch_gsm8k_h100.sh`: Launch on H100 GPUs
- `launch_gsm8k_h200.sh`: Launch on H200 GPUs
- `launch_gsm8k_b200.sh`: Launch on B200 GPUs
- `launch_gsm8k_mi300x.sh`: Launch on MI300X GPUs
- `launch_gsm8k_mi325x.sh`: Launch on MI325X GPUs
- `launch_gsm8k_mi355x.sh`: Launch on MI355X GPUs

These scripts handle:
- Docker container setup
- GPU configuration (CUDA/ROCm)
- Volume mounts
- Environment variable passing

## Usage

### Via GitHub Actions UI

1. Go to Actions tab
2. Select "GSM8k Benchmark" workflow
3. Click "Run workflow"
4. Fill in the parameters
5. Click "Run workflow"

### Example

To evaluate Llama-3.1-8B-Instruct on H100 with vLLM:
- model: `meta-llama/Llama-3.1-8B-Instruct`
- runner: `h100-cr`
- image: `vllm/vllm-openai:latest`
- framework: `vllm`
- precision: `fp16`
- tp: `1`

## Results

Results are saved as JSON files and uploaded as GitHub Actions artifacts. The output includes:
- Accuracy metrics
- Per-example results
- Model and configuration metadata

## Dependencies

The benchmark requires:
- `lm-eval[vllm]`: Language Model Evaluation Harness with vLLM support
- A running inference server (vLLM or SGLang)
- Access to the model on HuggingFace Hub

## References

- [GSM8k Paper](https://arxiv.org/abs/2110.14168)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
