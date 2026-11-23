# Build and run Strata Ampere on Windows

This guide builds the CUDA engine for NVIDIA `sm_86` GPUs and runs it with the native Strata
HTTP server. The example uses Qwen3.8-Flash-Next IQ3_S, one request slot, MTP, and the GPU
image encoder. See the [model guide](MODELS.md#qwen38-flash-next) for weight and memory
requirements.

## Build the CUDA engine

Use the x64 developer shell for Visual Studio 2022 with CMake, Ninja and the CUDA 13 toolkit
on `PATH`. Run the following commands from the source directory.

```powershell
cmake -S engine -B build-ampere -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DSTRATA_ENABLE_CUDA=ON `
  -DSTRATA_PORTABLE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF `
  -DSTRATA_BUILD_INKLING=OFF -DSTRATA_BUILD_INKLING2=OFF
cmake --build build-ampere --target strata --parallel 4
.\build-ampere\strata.exe --version
```

The executable identifies itself as `strata 0.1.66` and prints `strata local: deferred CPU experts`.
For the example profile, put `strata-vision.exe` beside it. The companion encoder is available
in the [Strata v0.2.20 Windows CUDA 13 engine archive](https://github.com/Niko1221/Strata/releases/download/v0.2.20/strata-engine-0.1.66-windows-x86_64-cuda-cuda13.tar.gz).
Keep the CUDA runtime DLLs available on `PATH`.

## Prepare a model profile

Use the [Strata model setup](AI_SETUP.md) to prepare an IQ3_S run profile with MTP and vision.
For the Terminal launcher, select `--family qwen --model iq3_s --drafting mtp --vision gpu`.
The [launcher reference](TERMINAL_LAUNCH.md) describes where it writes the run configuration.
The base profile supplies the pack, model shards, MTP weights, tokenizer, expert profile and
image projector.

Use the Python runtime from a Strata installation or install the dependencies in
[`terminal/requirements.txt`](../terminal/requirements.txt). Set `$BaseProfile` to the IQ3_S
profile and `$AmpereProfile` to a new absolute JSON path outside the checkout. From the source
directory, generate the Ampere example profile and start the server.

```powershell
$Engine = (Resolve-Path .\build-ampere\strata.exe).Path
python .\tools\ampere\prepare_profile.py --profile $BaseProfile --engine $Engine `
  --out $AmpereProfile --port 8080
Set-Location .\terminal
python -m serve.server --engine strata --config $AmpereProfile --host 127.0.0.1 --port 8080
```

The profile generator checks that the executable accepts its engine arguments. Review the
context length, KV cache settings, expert RAM budget and VRAM reserve in the generated file
against the available memory. For a first run with limited memory, reduce `--max-context`
and keep `--kv-resident` at or below that length.

Open `http://127.0.0.1:8080/#monitor` for the dashboard. The API is available at
`http://127.0.0.1:8080/v1`. See the [Strata API reference](DETAILS.md#using-it) for request examples.

## Control CPU expert deferral

| Profile environment | Effect |
| --- | --- |
| `STRATA_VERIFY_DEFER_CPU=1` | Enable deferral for eligible decode verification windows. |
| `STRATA_VERIFY_DEFER_CPU_BELOW=0.1` | Defer CPU contributions below this routing weight. Contributions at or above the threshold join in the current layer. |
| `STRATA_VERIFY_DEVICE_HITS=0`, `STRATA_VERIFY_DEVICE_PLAN=0` | Use host-side expert planning for the verifier in the example profile. |

The example uses one sequence and host-side expert planning. A fully VRAM-resident window
has no CPU contributions to defer. Batched verification and device-side expert planning
use the original path. Check the engine log for the configured threshold and a captured
verification graph marked `deferred`.

## Compare runs

Use the same build and set `STRATA_VERIFY_DEFER_CPU` to `0` for the baseline and `1` for the
comparison. Restart the model between settings. Keep the weights, expert placement, MTP
settings, prompt and output token budget, and sampling settings fixed. Warm up each run,
then repeat each prompt several times with the same prompt-cache policy.

Record prefill throughput, decode throughput, and time to first token separately.
Prefill reports the rate of reading the input. Decode reports the rate of generating
output. Deferral targets the decode step. Compare model outputs on the tasks you intend
to run as well as measuring speed, since changing the point where CPU contributions join
can change predictions.
