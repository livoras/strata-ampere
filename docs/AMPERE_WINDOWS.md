# Strata Ampere on Windows

This fork starts from [Strata v0.2.20](https://github.com/Niko1221/Strata/releases/tag/v0.2.20)
(engine 0.1.66). It adds the opt-in CPU expert deferral used on the Windows RTX 3090 reference machine.
During MTP verification, the engine delays CPU expert rows below the configured routing weight.
Enabling deferral can change model output.

The patch comes from lask3802's Strata
[#2689](https://github.com/Niko1221/Strata/pull/2689) and
[#2690](https://github.com/Niko1221/Strata/pull/2690). It follows the cold-batch port that
[Speedy's NOTICE](https://github.com/DevDock-AI/Speedy/blob/v0.5.4/NOTICE)
lists among its ports of MIT-licensed Strata contributions.

## Build the CUDA engine

Use the x64 developer shell for Visual Studio 2022 with CMake, Ninja and the CUDA toolkit on
`PATH`. The reference build used MSVC 19.44.35230 and CUDA 13.3.73. Run these commands in
the fork's source directory.

```powershell
cmake -S engine -B build-ampere -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DSTRATA_ENABLE_CUDA=ON `
  -DSTRATA_PORTABLE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF `
  -DSTRATA_BUILD_INKLING=OFF -DSTRATA_BUILD_INKLING2=OFF
cmake --build build-ampere --target strata --parallel 4
.\build-ampere\strata.exe --version
```

The command should print `strata 0.1.66` and `strata local: deferred CPU experts`. For image
input, put `strata-vision.exe` from the official
`strata-engine-0.1.66-windows-x86_64-cuda-cuda13.tar.gz` next to the custom `strata.exe`.
The upstream archive's SHA-256 is
`d0fa583e60aaf1d256926b6ea9889ff3ce42a98dbf56f2b556703a76b65e0a64`.
Keep the CUDA 13 DLLs used by the build available on `PATH`.

## Run the native Strata API

Start with a Qwen3.8-Flash-Next IQ3_S profile with MTP and vision. Use the Strata
[model setup](AI_SETUP.md) to prepare the assets and select MTP with `--drafting mtp`.
The base profile supplies the local pack, MTP, expert profile, tokenizer and vision
model paths. The tool writes a new machine-local profile and checks that the custom
engine accepts the arguments. Keep model assets available at the paths in the new
profile.

Use Python from an existing Strata installation or install the dependencies in
`terminal/requirements.txt`. Set `$BaseProfile` to the existing model profile and
`$AmpereProfile` to an absolute path for a new JSON file outside the checkout.
From the fork's root:

```powershell
$Engine = (Resolve-Path .\build-ampere\strata.exe).Path
python .\tools\ampere\prepare_profile.py --profile $BaseProfile --engine $Engine `
  --out $AmpereProfile --port 8080
Set-Location .\terminal
python -m serve.server --engine strata --config $AmpereProfile --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/#monitor` for the built-in v0.2.20 dashboard.
For clients that still send an older model ID, include its name in the profile's
`aliases` list. The reference Codex bridge sends
`qwen3.8-flash-next-unsloth-ud-iq2_xs` as a legacy routing name. The loaded model
is `qwen3.8-flash-next-iq3_s` through either name.

The reference profile uses one request slot, a 262,144-token context, INT8 KV with
32,768 positions per layer in VRAM, a 1,600 MiB VRAM reserve, a 38 GiB expert resident
budget, and the vision encoder on the GPU.
It sets the deferral threshold to `0.1` and selects host-side expert planning for
the verifier. Check the engine log for `STRATA_VERIFY_DEFER_CPU=1` and a captured
graph marked `deferred`.

## Reference validation

All nine changed engine files matched the Windows build tree by SHA-256. The reference
Windows executable has SHA-256
`971ab0ba6c6ceecd858e5e8d3f40df4d1dc5e45a15849ec68dfa1c85e15ff680`.

The reference Windows RTX 3090 served the same three streaming text requests through
the Speedy 0.5.4 and native Strata APIs, with the same engine executable, model,
profile, and client entry point. Each request generated 256 tokens and reused zero
prompt tokens. Prefill and decode are in tokens per second, and time to first token is
measured at the client in seconds.

| Prompt tokens | Prefill Speedy → Strata | Decode Speedy → Strata | First token Speedy → Strata |
| ---: | ---: | ---: | ---: |
| 4,255 | 1,379.1 → 1,430.5 | 91.8 → 94.5 | 3.160 → 3.048 |
| 9,155 | 1,404.2 → 1,379.2 | 51.7 → 64.0 | 6.723 → 6.772 |
| 14,755 | 1,704.9 → 1,683.9 | 83.8 → 89.0 | 8.757 → 8.867 |

There was one sample per length and API. The answer text and expert hit rates varied
between runs. All six measured requests logged zero expert file reads. Native Strata
also passed the GPU image, Anthropic Messages and OpenAI Responses tool-call checks.

The [first snapshot](https://github.com/livoras/strata-ampere/tree/v0.2.20-ampere.1)
documents the earlier deployment with the Speedy v0.5.4 API and the same engine.
