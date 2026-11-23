"""Make a local Speedy 0.5.4 profile for the tested Strata Ampere build."""

import argparse
import json
import os
import re
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


def model_file(args, flag):
    if flag not in args:
        raise ValueError(f"Missing {flag} in the source profile")
    index = args.index(flag) + 1
    if index == len(args) or not args[index] or args[index].startswith("--"):
        raise ValueError(f"Missing file argument for {flag}")
    return args[index]


def prepare(source, engine, output, port):
    profile = json.loads(source.read_text(encoding="utf-8-sig"))
    if profile.get("model_name") != MODEL or profile.get("drafting") != "mtp":
        raise ValueError(f"Use a Speedy profile for {MODEL} with MTP and vision")
    vision = profile.get("vision")
    if not isinstance(vision, dict) or not vision.get("model") or not vision.get("mmproj"):
        raise ValueError("The source profile must include the vision model and projector")
    encoder = engine.with_name("strata-vision.exe")
    if not encoder.is_file():
        raise ValueError("Place Strata 0.1.66 strata-vision.exe beside the custom engine")
    vision.update(exe=str(encoder), gpu=True, max_tokens=1024, elastic=False)

    args = profile["args"]
    profile.update(
        exe=str(engine), lib_dirs=[str(engine.parent)], gpu=0, parallel=1,
        env=ENGINE_ENV, engine_ready_s=946, host="127.0.0.1", port=port,
        log=str(output.with_suffix(".log")), open_browser=False,
        args=[
            "--pack", model_file(args, "--pack"),
            "--native", model_file(args, "--native"),
            "--expert-profile", model_file(args, "--expert-profile"),
            "--expert-cache", "auto", "--prefill", "auto",
            "--spec", "4", "--spec-min-p", "0.5", "--mtp", model_file(args, "--mtp"),
            "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768",
            "--vision", "--vram-reserve-mib", "1600", "--resident-budget-gib", "38",
            "--conversation-cache-mib", "0",
            "--ple-gguf", model_file(args, "--ple-gguf"), "--ple-io", "direct",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--speedy-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a local port between 1024 and 65535")

    try:
        version_file = args.speedy_root / "serve" / "version.py"
        if not re.search(r'^SPEEDY_VERSION\s*=\s*"0\.5\.4"\s*$', version_file.read_text(encoding="utf-8"), re.M):
            raise ValueError("Use Speedy 0.5.4")
        source, engine, output = (path.resolve() for path in (args.profile, args.engine, args.out))
        profile = prepare(source, engine, output, args.port)
        check_engine(profile)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)
            handle.write("\n")
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{exc}\n")
    print(output)


if __name__ == "__main__":
    main()
