# Strata Ampere

Strata Ampere is an independent source project for running large mixture-of-experts (MoE)
models on Windows with an NVIDIA Ampere GPU. It combines the native HTTP server and Monitor
from [Strata v0.2.20](https://github.com/Niko1221/Strata/releases/tag/v0.2.20) with an
opt-in change to the CUDA decode verifier in engine 0.1.66.

## Why this project exists

MoE models select a small set of expert networks for each token. A single GPU can keep some
of those experts in VRAM, while Strata can run other selected experts on the CPU. During
decoding, the GPU can spend part of each layer waiting for the CPU results. Strata Ampere
targets that wait on machines that split expert computation between CPU and GPU.

## How it works

With multi-token prediction (MTP) enabled, a draft layer proposes tokens for the main model
to verify. When the verifier can split the CPU work by routing weight, it computes the
higher-weight contributions first and combines them in the current layer. It lets the CPU finish the
lower-weight contributions while the GPU starts the next layer's attention or state mixer.
The delayed contributions join the residual stream before the next layer's feed-forward
block. The overlap can shorten the wait for CPU experts.

The next mixer sees the residual before those contributions have been added, so enabling
deferral can change model output. The gain depends on how much expert work lands on the CPU,
the model and the workload. The feature is off by default in the engine. The Windows example
profile turns it on with a routing-weight threshold of `0.1`.

The HTTP server formats and tokenizes requests, runs the CUDA engine as a local process, and
streams its output to the client. It provides OpenAI-compatible Chat Completions and Responses,
Anthropic-compatible Messages, and a built-in Monitor. The expert scheduling change runs inside
the engine process.

## Build, run and measure

The [Windows guide](docs/AMPERE_WINDOWS.md) covers the CUDA `sm_86` build, the Qwen3.8-Flash-Next
IQ3_S example, and a way to compare decoding with deferral on and off. See the
[Strata API and engine reference](docs/DETAILS.md) for the shared API and runtime settings.

## Attribution and license

The base engine and server come from [Strata](https://github.com/Niko1221/Strata). CPU expert
deferral comes from lask3802's Strata [#2689](https://github.com/Niko1221/Strata/pull/2689)
and [#2690](https://github.com/Niko1221/Strata/pull/2690). The implementation includes the
cold-batch adaptation credited in the
[Speedy NOTICE](https://github.com/DevDock-AI/Speedy/blob/v0.5.4/NOTICE).

The project is distributed under the [MIT License](LICENSE). Bundled third-party components
carry their own license notices.
