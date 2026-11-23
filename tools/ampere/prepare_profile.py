"""Make a native Strata profile for the tested Windows Ampere build."""

import argparse
import json
import os
import subprocess
from pathlib import Path


MODEL = "qwen3.8-flash-next-iq3_s"
ENGINE_ENV = {
    "STRATA_RESIDENT_HEADROOM_GIB": "4",
    "STRATA_ROUTE_TAIL_SKIP": "0",
    "STRATA_PF_GROUP_SCRATCH": "1",
    "STRATA_PF_HC_REGION": "1",
    "STRATA_KV_PREFETCH": "1",
    "STRATA_VERIFY_DEFER_CPU": "1",
    "STRATA_VERIFY_DEFER_CPU_BELOW": "0.1",
    "STRATA_VERIFY_DEVICE_HITS": "0",
    "STRATA_VERIFY_DEVICE_PLAN": "0",
}


def local_path(cwd, value):
    return str((cwd / value).resolve())


def model_file(args, flag, cwd):
    if flag not in args:
        raise ValueError(f"Missing {flag} in the source profile")
    index = args.index(flag) + 1
    if index == len(args) or not args[index] or args[index].startswith("--"):
        raise ValueError(f"Missing file argument for {flag}")
    return local_path(cwd, args[index])


def prepare(source, engine, output, port, *, long_prefill=False, spec_min_p=0.5):
    profile = json.loads(source.read_text(encoding="utf-8-sig"))
    if profile.get("model_name") != MODEL or profile.get("drafting") != "mtp":
        raise ValueError(f"Use a Qwen profile for {MODEL} with MTP and vision")
    vision = profile.get("vision")
    if not isinstance(vision, dict) or not vision.get("model") or not vision.get("mmproj"):
        raise ValueError("The source profile must include the vision model and projector")
    encoder = engine.with_name("strata-vision.exe")
    if not encoder.is_file():
        raise ValueError("Place Strata 0.1.66 strata-vision.exe beside the custom engine")
    cwd = source.parent / (profile.get("cwd") or ".")
    vision.update(exe=str(encoder), gpu=True, max_tokens=1024, elastic=False,
                  model=local_path(cwd, vision["model"]), mmproj=local_path(cwd, vision["mmproj"]))

    args = profile["args"]
    env = ENGINE_ENV.copy()
    if long_prefill:
        env.update(STRATA_VERIFY_DEFER_CPU="0", STRATA_ROUTE_PRIOR="0", STRATA_KVA="0")
    profile.update(
        exe=str(engine), cwd=str(engine.parent), lib_dirs=[str(engine.parent)], gpu=0, parallel=1,
        tokenizer=local_path(cwd, profile["tokenizer"]),
        env=env, engine_ready_s=946, host="127.0.0.1", port=port,
        log=str(output.with_suffix(".log")), open_browser=False,
        args=[
            "--pack", model_file(args, "--pack", cwd),
            "--native", model_file(args, "--native", cwd),
            "--expert-profile", model_file(args, "--expert-profile", cwd),
            "--expert-cache", "auto", "--prefill", "auto:16384" if long_prefill else "auto",
            "--spec", "4", "--spec-min-p", str(spec_min_p), "--mtp", model_file(args, "--mtp", cwd),
            "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768",
            "--vision", "--vram-reserve-mib", "1600", "--resident-budget-gib", "38",
            "--conversation-cache-mib", "0",
            "--ple-gguf", model_file(args, "--ple-gguf", cwd), "--ple-io", "direct",
            "--adapt-async", "1", "--adapt-swaps", "80", "--pool-affinity", "measure",
        ],
    )
    return profile


def check_engine(profile):
    env = dict(os.environ, **profile["env"])
    env["PATH"] = os.pathsep.join([*profile["lib_dirs"], env.get("PATH", "")])
    exe = profile["exe"]
    version = subprocess.run([exe, "--version"], env=env, text=True, errors="replace",
                             capture_output=True, timeout=15, check=True)
    expected = {"strata 0.1.66", "strata local: deferred CPU experts"}
    if not expected.issubset(version.stdout.splitlines()):
        raise ValueError("Use the Strata Ampere 0.1.66 build")
    # The final argument makes the parser exit before loading weights.
    marker = "--strata-ampere-profile-probe"
    result = subprocess.run([exe, "--serve", *profile["args"], marker], env=env,
                            text=True, errors="replace", capture_output=True, timeout=15)
    if result.returncode != 2 or f"unknown argument: {marker}" not in result.stderr + result.stdout:
        raise ValueError("The engine rejected the Windows profile arguments")


def edit_options(args, options):
    result = []
    index = 0
    while index < len(args):
        flag = args[index]
        if flag not in options:
            result.append(flag)
            index += 1
            continue
        if index + 1 == len(args) or args[index + 1].startswith("--"):
            raise ValueError(f"Missing value for {flag} in the source profile")
        index += 2
    for flag, value in options.items():
        if value is not None:
            result.extend([flag, str(value)])
    return result


def prepare_candidate(source, engine, port, draft_model, draft_vocab, cuda_bin,
                      *, resident_budget=40, vram_reserve=1200):
    profile = json.loads(source.read_text(encoding="utf-8-sig"))
    if profile.get("model_name") != MODEL or not (profile.get("vision") or {}).get("gpu"):
        raise ValueError(f"Use the Windows Ampere profile for {MODEL} with CUDA vision")
    if not draft_model.is_file() or not draft_vocab.is_file():
        raise ValueError("Supply the local DFlash2 Q4_0 pack and draft vocabulary")
    options = {
        "--mtp": None, "--spec-min-p": None, "--suffix-draft": None,
        "--draft": "dflash2", "--draft-model": draft_model, "--draft-fallback": "none",
        "--mtp-draft-vocab": draft_vocab, "--spec": 8, "--draft-conf": "0.60",
        "--pcie-frac": "0.20", "--prefill": "auto:16384",
        "--resident-budget-gib": resident_budget, "--vram-reserve-mib": vram_reserve,
        "--max-context": 262144, "--kv": "int8", "--kv-resident": 32768,
        "--conversation-cache-mib": 3072, "--conversation-spill-mib": 0,
        "--conversation-cache-min-free-mib": 2560,
    }
    profile["args"] = edit_options(profile["args"], options)
    # Control resolves DFlash2 defaults at load time. These three explicit overrides
    # keep the tested pack, window and confidence when the engine is reloaded.
    overrides = {flag: value if flag in ("--draft-model", "--spec", "--draft-conf") else None
                 for flag, value in options.items()}
    profile["user_args"] = edit_options(profile.get("user_args") or [], overrides)
    env = profile.setdefault("env", {})
    env.pop("STRATA_PREFILL_RING", None)
    env.update(STRATA_PREFILL_RING_LARGE="384", STRATA_PREFILL_CPU_SHARE_MAX="6144",
               STRATA_RESIDENT_HEADROOM_GIB="2", STRATA_VERIFY_DEFER_CPU="0",
               STRATA_VERIFY_DEVICE_HITS="0", STRATA_VERIFY_DEVICE_PLAN="0",
               STRATA_ROUTE_TAIL_SKIP="0", STRATA_ROUTE_PRIOR="0", STRATA_KVA="0")
    profile.update(exe=str(engine), port=port, drafting="dflash2")
    if cuda_bin:
        if not cuda_bin.is_dir():
            raise ValueError("The candidate CUDA runtime directory does not exist")
        profile["lib_dirs"] = list(dict.fromkeys([str(cuda_bin), *profile.get("lib_dirs", [])]))
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--long-prefill", action="store_true",
                        help="Allow prefill chunks up to 16,384 tokens and disable CPU expert deferral")
    parser.add_argument("--spec-min-p", type=float, default=0.5,
                        help="Minimum MTP confidence to extend a draft window (default: 0.5)")
    parser.add_argument("--candidate-2b-df2", action="store_true",
                        help="Apply the validated DFlash2 settings to a Windows serving profile")
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--draft-vocab", type=Path)
    parser.add_argument("--cuda-bin", type=Path)
    parser.add_argument("--resident-budget-gib", type=int, choices=(39, 40), default=40)
    parser.add_argument("--vram-reserve-mib", type=int, choices=(1200, 1400), default=1200)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a local port between 1024 and 65535")
    if not 0.0 <= args.spec_min_p <= 1.0:
        parser.error("--spec-min-p must be between 0 and 1")
    if args.candidate_2b_df2 and not (args.draft_model and args.draft_vocab):
        parser.error("--candidate-2b-df2 requires --draft-model and --draft-vocab")

    try:
        source, engine, output = (path.resolve() for path in (args.profile, args.engine, args.out))
        if args.candidate_2b_df2:
            profile = prepare_candidate(
                source, engine, args.port, args.draft_model.resolve(), args.draft_vocab.resolve(),
                args.cuda_bin.resolve() if args.cuda_bin else None,
                resident_budget=args.resident_budget_gib, vram_reserve=args.vram_reserve_mib,
            )
        else:
            profile = prepare(source, engine, output, args.port, long_prefill=args.long_prefill,
                              spec_min_p=args.spec_min_p)
        check_engine(profile)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)
            handle.write("\n")
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{exc}\n")
    print(output)


if __name__ == "__main__":
    main()
