import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import sys
import re

def run_job(inp_path: Path, exe_dir, wine=False, trv_name="trvmc95pc.exe", trim_name="TrimSPAuswertung.exe"):
    stem = inp_path.stem
    workdir = inp_path.parent / stem
    workdir.mkdir(exist_ok=True)

    # copy original .inp into working dir and name trvmc95.inp (like your batch did)
    target_trv_inp = workdir / "trvmc95.inp"
    shutil.copy2(inp_path, target_trv_inp)

    # Determine executable paths (exe_dir may be a Path (native) or a Windows-style string when using wine)
    if isinstance(exe_dir, str):
        # exe_dir is a Windows-style path provided while running on Linux (use wine)
        trv_path_str = f"{exe_dir}\\{trv_name}"
        trim_path_str = f"{exe_dir}\\{trim_name}"
    elif isinstance(exe_dir, Path):
        trv_path = exe_dir / trv_name
        trim_path = exe_dir / trim_name
        trv_path_str = str(trv_path)
        trim_path_str = str(trim_path)
    else:
        trv_path = Path(trv_name)
        trim_path = Path(trim_name)
        trv_path_str = str(trv_path)
        trim_path_str = str(trim_path)

    # Windows-only priority flags (safe to ignore on POSIX)
    creationflags = None
    startupinfo = None
    if os.name == "nt":
        try:
            creationflags = 0x00000080  # HIGH_PRIORITY_CLASS
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 6  # SW_MINIMIZE
            startupinfo = si
        except Exception:
            creationflags = None
            startupinfo = None

    def _run(cmd_list, cwd, env=None):
        logpath = Path(cwd) / f"{stem}.log"
        # include existing env plus any supplied
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        # capture output so we can inspect MATLAB errors
        proc = subprocess.run(cmd_list, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env, text=True)
        try:
            with open(logpath, "w", encoding="utf-8") as f:
                f.write(proc.stdout or "")
        except Exception:
            pass
        return proc.returncode, str(logpath)

    # 1) run trvmc95pc.exe (wait)
    # check existence: if using wine, we can't check local FS, assume user provided correct Windows path
    if not wine:
        if not Path(trv_path_str).exists() and shutil.which(trv_path_str) is None:
            return (stem, False, f"{trv_path_str} not found")
    try:
        cmd = (["wine", trv_path_str] if wine else [trv_path_str])
        rc, log = _run(cmd, workdir)
        if rc != 0:
            return (stem, False, f"trv exited {rc}; see {log}")
    except Exception as e:
        return (stem, False, f"trv run failed: {e}")

    # rename trvmc95 outputs to match original stem
    try:
        out_inp = workdir / f"{stem}.inp"
        out_out = workdir / f"{stem}.out"

        # replace (overwrite) if destination already exists
        if target_trv_inp.exists():
            try:
                os.replace(str(target_trv_inp), str(out_inp))
            except Exception:
                # fallback: remove existing target then rename
                if out_inp.exists():
                    out_inp.unlink()
                target_trv_inp.rename(out_inp)

        trv_out = workdir / "trvmc95.out"
        if trv_out.exists():
            try:
                os.replace(str(trv_out), str(out_out))
            except Exception:
                if out_out.exists():
                    out_out.unlink()
                trv_out.rename(out_out)
    except Exception as e:
        return (stem, False, f"rename after trv failed: {e}")

    # 2) run TrimSPAuswertung.exe -q fort.17 "<stem>"
    if not wine:
        if not Path(trim_path_str).exists() and shutil.which(trim_path_str) is None:
            return (stem, False, f"{trim_path_str} not found")
    try:
        cmd = (["wine", trim_path_str, "-q", "fort.17", stem] if wine else [trim_path_str, "-q", "fort.17", stem])
        # isolate MATLAB environment so user startup.m (which does cd('K:\...')) is not run:
        # - set a dedicated MATLAB_PREFDIR
        # - override HOME/USERPROFILE so MATLAB's default userpath points into the workdir
        # these are minimal, per-process overrides; adjust if your environment needs others removed.
        (workdir / "matlab_pref").mkdir(exist_ok=True)
        (workdir / "matlab_user").mkdir(exist_ok=True)
        overrides = {"MATLAB_PREFDIR": str(workdir / "matlab_pref")}
        if os.name == "nt":
            overrides["USERPROFILE"] = str(workdir / "matlab_user")
        else:
            overrides["HOME"] = str(workdir / "matlab_user")
        # also avoid inheriting any custom MATLABPATH
        overrides.pop("MATLABPATH", None)
        rc, log = _run(cmd, workdir, env=overrides)
        if rc != 0:
            return (stem, False, f"trim exited {rc}; see {log}")
    except Exception as e:
        return (stem, False, f"trim run failed: {e}")

    # move fort.17 -> <stem>.fort.17
    try:
        fort = workdir / "fort.17"
        if fort.exists():
            dest = workdir / f"{stem}.fort.17"
            try:
                # atomic replace where supported, avoids WinError 183 if dest exists
                os.replace(str(fort), str(dest))
            except Exception:
                # fallback: remove existing destination then rename
                if dest.exists():
                    dest.unlink()
                fort.rename(dest)
    except Exception as e:
        return (stem, False, f"move fort failed: {e}")

    return (stem, True, "ok")


def main():
    # use CPU core count as default worker count
    default_workers = max(1, os.cpu_count() or 1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", "-d", default=".", help="directory with .inp files (default cwd)")
    ap.add_argument("--exe-dir", "-e", default=None, help="directory containing the .exe files (default: PATH or TRIM.SP default on Windows)")
    ap.add_argument("--workers", "-w", type=int, default=default_workers, help=f"parallel workers (default: number of CPU cores: {default_workers})")
    ap.add_argument("--pattern", "-p", default="*.inp", help="glob pattern for input files")
    args = ap.parse_args()

    # default to the script's folder when the user left --dir as "."
    if args.dir == ".":
        base = Path(__file__).parent.resolve()
    else:
        base = Path(args.dir).resolve()
    # If running on Windows and no exe-dir passed, default to TRIM.SP install folder
    if args.exe_dir is None and os.name == "nt":
        exe_dir = Path(r"C:\Program Files (x86)\TRIM.SP")
    else:
        # If user passed a Windows-style path but we're on Linux, keep it as string and enable wine
        if args.exe_dir and os.name != "nt" and re.match(r"^[A-Za-z]:\\", args.exe_dir):
            exe_dir = args.exe_dir  # keep as Windows-style string for wine
        else:
            exe_dir = Path(args.exe_dir).resolve() if args.exe_dir else None

    wine = (os.name != "nt" and isinstance(exe_dir, str))

    inps = sorted(base.glob(args.pattern))
    if not inps:
        print("No .inp files found.", file=sys.stderr)
        sys.exit(1)

    # summarize parallel execution plan
    trv_name = "trvmc95pc.exe"
    trim_name = "TrimSPAuswertung.exe"
    if isinstance(exe_dir, str):
        trv_path_str = f"{exe_dir}\\{trv_name}"
        trim_path_str = f"{exe_dir}\\{trim_name}"
    elif isinstance(exe_dir, Path):
        trv_path_str = str(exe_dir / trv_name)
        trim_path_str = str(exe_dir / trim_name)
    else:
        trv_path_str = trv_name
        trim_path_str = trim_name

    workers_requested = args.workers
    concurrency = min(workers_requested, len(inps))
    print(f"Workers requested: {workers_requested}")
    print(f"Input jobs found: {len(inps)}")
    print(f"Estimated parallel concurrency: {concurrency}")
    print(f"TRV executable: {trv_path_str}{' (via wine)' if wine else ''}")
    print(f"TRIM executable: {trim_path_str}{' (via wine)' if wine else ''}")
    print("Job steps (per input): copy <stem>.inp -> <stem>/trvmc95.inp; run TRV; rename outputs; run TRIM (-q fort.17 <stem>); move fort.17 -> <stem>.fort.17")
    print("Jobs to run:")
    for p in inps:
        print("  -", p.name)

    # show progress bar if tqdm is installed, otherwise a simple counter
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    results = []
    total = len(inps)
    pbar = tqdm(total=total, desc="Jobs", unit="job") if tqdm else None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_job, p, exe_dir, wine): p for p in inps}
        completed = 0
        for fut in as_completed(futures):
            stem, ok, msg = fut.result()
            status = "OK" if ok else "FAIL"
            print(f"{stem}: {status} - {msg}")
            results.append((stem, ok, msg))
            completed += 1
            if pbar:
                pbar.update(1)
            else:
                print(f"Progress: {completed}/{total} jobs finished")
    if pbar:
        pbar.close()

    fails = [r for r in results if not r[1]]
    if fails:
        print(f"{len(fails)} jobs failed", file=sys.stderr)
        sys.exit(2)
    print("All jobs finished successfully.")


if __name__ == "__main__":
    main()
    if os.name == "nt" and sys.stdin.isatty() is False:
        # Only prompt if running in a Windows console that will close
        input("Press Enter to exit...")
