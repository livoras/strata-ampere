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

### Read long inputs in larger chunks

Pass `--long-prefill` to `prepare_profile.py` and give `--out` a new profile path.
This sets `--prefill auto:16384` and `STRATA_VERIFY_DEFER_CPU=0` so the verifier waits for
all CPU expert contributions. Larger chunks can reduce repeated expert transfers during prefill.
They also borrow more GPU cache slots. Check the engine log for the actual chunk size and for
expert weights read from disk when RAM cannot back all borrowed slots.

### Tune the decode draft window

Use both options when generating the performance profile from the `prefill-decode` branch.

```powershell
python .\tools\ampere\prepare_profile.py --profile $BaseProfile --engine $Engine `
  --out $AmpereProfile --port 8080 --long-prefill --spec-min-p 0.65
```

The `--spec-min-p 0.65` option raises the MTP confidence threshold.
The draft layer stops extending a window sooner when it is unsure of its next prediction.
The main model still verifies each token. Compare decode speed and output quality across
several warm prompts with the same cache and sampling settings.

### Reproduce Candidate-2B-DF2

Build commit `071ff14b7` and install its `strata.exe` in a versioned directory beside
the original engine. Use a separate executable when more than one profile shares the
original engine. Keep the existing CUDA vision executable and CUDA runtime in place.
Generate the new profile from the active Windows Ampere IQ3_S profile so its aliases,
vision settings, asset paths and service settings carry over.

```powershell
python .\tools\ampere\prepare_profile.py --candidate-2b-df2 --profile $ActiveProfile --engine $CandidateEngine --draft-model $DFlash2Q4Pack --draft-vocab $DraftVocab --cuda-bin $CudaBin --out $CandidateProfile
```

Use the DFlash2 Q4_0 pack. The helper pins the draft window to 8, confidence to 0.60
and PCIe share to 0.20. It keeps the engine's default suffix minimum of 3. The
profile selects `auto:16384` prefill, a 384-slot ring for large chunks and CPU
expert sharing through 6,144 tokens. It uses 40 GiB for resident experts, a 2 GiB
resident headroom and a 1,200 MiB VRAM reserve. For the 256K service it keeps
int8 KV with 32,768 resident tokens, a 3,072 MiB RAM conversation cache, a
2,560 MiB cache memory floor and zero SSD spill. CPU verify deferral, route
skipping, route priors and KVA are disabled.

Back up the active profile and executable with their SHA-256 hashes. Wait for
the API's in-flight and queued work to reach zero, then stop the serving task
and verify its loopback listener and vision child have exited. Put the staged
profile at the active profile path and start the same task. Restore the saved
profile with the same drain-and-restart procedure to roll back.

Check the effective DFlash2 and prefill settings in the startup log, and verify
the original model alias through the client endpoint. Run text, image and
cross-conversation cache probes. Measure physical and commit headroom and GPU
memory at load and after replay. The profile helper accepts
`--resident-budget-gib 39` and `--vram-reserve-mib 1400` for memory-pressure
retests. Keep the Windows power plan and GPU power limit at their operating
settings during comparisons.

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
