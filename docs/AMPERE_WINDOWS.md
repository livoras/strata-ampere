# Strata Ampere on Windows

This fork starts from [Strata v0.2.20](https://github.com/Niko1221/Strata/releases/tag/v0.2.20)
(engine 0.1.66). It adds the opt-in CPU expert deferral used on the Windows RTX 3090 reference machine.
During MTP verification, the engine delays CPU expert rows below the configured routing weight.
Enabling deferral can change model output.

The patch comes from lask3802's Strata
[#2689](https://github.com/Niko1221/Strata/pull/2689) and
[#2690](https://github.com/Niko1221/Strata/pull/2690). It follows the cold-batch port that
[Speedy's NOTICE](https://github.com/DevDock-AI/Speedy/blob/v0.5.4/NOTICE)
lists among its ports of MIT-licensed Strata contributions. The Monitor patch updates three
web assets using code from the Strata v0.2.7.1 and v0.2.20 releases.

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

## Reproduce the reference service

The reference service uses [Speedy v0.5.4](https://github.com/DevDock-AI/Speedy/releases/tag/v0.5.4)
as a separate installation. Start with a Speedy profile for the Qwen3.8-Flash-Next IQ3_S
pack with MTP and vision. The profile supplies the local pack, MTP, expert profile,
tokenizer and vision model paths. The tool copies these into a new machine-local profile
and verifies that the custom engine accepts the resulting arguments.

Set `$SpeedyRoot` to the Speedy v0.5.4 source directory, `$BaseProfile` to the model
profile and `$AmpereProfile` to a new JSON path outside the checkout. From the fork's root:

```powershell
$Engine = (Resolve-Path .\build-ampere\strata.exe).Path
$MonitorPatch = (Resolve-Path .\tools\ampere\speedy-v0.5.4-monitor.patch).Path
python .\tools\ampere\prepare_profile.py --profile $BaseProfile --engine $Engine `
  --speedy-root $SpeedyRoot --out $AmpereProfile --port 18080
git -C $SpeedyRoot apply --check $MonitorPatch
git -C $SpeedyRoot apply $MonitorPatch
Set-Location $SpeedyRoot
python -m serve.server --engine strata --config $AmpereProfile --host 127.0.0.1 --port 18080
```

Open `http://127.0.0.1:18080/#monitor` for the updated dashboard.
The reference profile uses one request slot, a 262,144-token context, INT8 KV with
32,768 positions per layer in VRAM, a 1,600 MiB VRAM reserve, a 38 GiB expert resident
budget, and the vision encoder on the GPU.
It sets the deferral threshold to `0.1` and selects host-side expert planning for
the verifier. Check the engine log for `STRATA_VERIFY_DEFER_CPU=1` and a captured
graph marked `deferred`.

## Reference validation

All nine changed engine files matched the Windows build tree by SHA-256. Applying the
Monitor patch to the Speedy v0.5.4 source produced the same three web files as the
Windows service, also verified by SHA-256. The profile tool passed the argument probe
on the reference executable and reproduced the service's engine arguments and
environment. The reference Windows executable has SHA-256
`971ab0ba6c6ceecd858e5e8d3f40df4d1dc5e45a15849ec68dfa1c85e15ff680`.

On that Windows RTX 3090 service, a request with 9,152 fresh input tokens and 256 output
tokens measured 1,493 tok/s prefill, 74.7 tok/s decode and 6.224 seconds to first token.
The service passed text, image and tool-call checks. These results describe that run
with the stated model and profile.
