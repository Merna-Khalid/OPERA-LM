"""patha_session.py -- the Path A study session engine for Kaggle.

One entry point per session. It reads the persistent state file, picks
the longest-unfinished stage, runs it inside a wall-clock governor, and
pushes the checkpoint dataset before exiting -- whether the stop is the
governor's clean stop or the hard-deadline watchdog. Every training run
is launched with the FULL total step target and --resume: the WSD decay
window stays anchored to the true total across sessions, and the exact-
stream resume machinery in opera_lm.train / train_chat.py makes a
resumed session's batch stream identical to an uninterrupted one.

Stages (state-tracked, auto-ordered):
  env      pip deps + opera_lm selftest + triton kernel --test
  data     tokenizer 2M lines (+ saturation check per prereg §5.1),
           smoltalk + fineweb pools at T=512/8192, pack to mmap pools,
           bucket-population report; pushes the DATA dataset once
  smoke    20M/500-step end-to-end (packed source, 8k eval, position
           curve) + 155M throughput measurements (OPERA 2xT4 DDP, TF
           batch-32) -> s/step recorded in state
  pretrain_opera   FineWeb-Edu, target 15000 steps (prereg §2.3), DDP
  pretrain_tf      rope + nope concurrently, one per T4, batch 32
  sft_opera        smoltalk, 2500 steps, --init-weights-from
  sft_tf           rope + nope SFT pair (or single if nope lags)
  curves           position curves x3 to 8192, gates H1-H4, samples

Arms: opera (d=1664 nb=416 L=8, DDP 2xT4, batch 16/rank -> eff. 32),
rope / nope (d=1264 7L, single T4, batch 32 -> eff. 32). Matched per
docs/OPERA_PathA_prereg.md §2.2.

Usage (repo root, inside the Kaggle notebook):
  python kaggle/patha_session.py                 # auto: next stage(s)
  python kaggle/patha_session.py --stages data   # force a stage
  python kaggle/patha_session.py --stages status # print state, exit

Env: KAGGLE_USERNAME/KAGGLE_KEY (notebook secrets), GOVERNOR_HOURS
(default 10.5; hard stop 30 min later, under Kaggle's 12h session cap),
SKIP_SELFTEST=1 to skip the full selftest in `env`.
"""
import argparse
import glob
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
WORK = os.environ.get("PATHA_WORK", "/kaggle/working/patha")
WORK_DATA = os.environ.get("PATHA_WORK_DATA", "/kaggle/working/patha_data")
DATA_IN = os.environ.get("PATHA_DATA_IN", "/kaggle/input/opera-lm-patha-data")
CKPT_IN = os.environ.get("PATHA_CKPT_IN", "/kaggle/input/opera-lm-patha-ckpt")
GOVERNOR_HOURS = float(os.environ.get("GOVERNOR_HOURS", "10.5"))
HARD_MARGIN_MIN = float(os.environ.get("PATHA_HARD_MARGIN_MIN", "30"))

# Protocol constants (docs/OPERA_PathA_prereg.md §2; do not change
# without an entry in the prereg's amendments log).
MAX_LEN, EVAL_MAX = 512, 8192
PRETRAIN_STEPS, SFT_STEPS = 15000, 2500
SAVE_EVERY = 1000
OPERA_ARGS = ["--d", "1664", "--nb", "416", "--num-layers", "8"]
SMOKE_OPERA_ARGS = ["--d", "640", "--nb", "160", "--num-layers", "4"]
TF_ARGS = ["--d", "1264", "--num-layers", "7"]
SMOKE_TF_ARGS = ["--d", "512", "--num-layers", "4"]      # ~21M (4.7 pilot scale)
SWEEP_STEPS = 2000      # 20M recipe-sweep length (curriculum every 50)
PROBE_STEPS = 250       # 155M confirmation probe length
# Recipe defaults (prereg amendment 2026-09-09): fusion_gate routing is a
# correctness fix (measured misallocation), wd 0.01 follows Moonshot +
# the byte rung; both + the lr are RE-MEASURED by the Session-2 sweep on
# this study's own data/hardware, with pre-stated adoption rules.
RECIPE_DEFAULT = {"opera": {"lr": "0.02", "wd": "0.01",
                            "include": "fusion_gate"},
                  "tf": {"lr": "0.02", "wd": "0.01"}}


def get_recipe(st):
    return st.get("recipe") or RECIPE_DEFAULT


def _final_loss(out_dir, log_name=None):
    """Final training loss of a probe run: last results row that has one,
    else the last 'loss <x>' step print in the run's log."""
    import glob as _glob
    import json as _json
    for rj in sorted(_glob.glob(os.path.join(out_dir, "*results.jsonl"))):
        rows = [_json.loads(x) for x in open(rj) if x.strip()]
        for r in reversed(rows):
            v = r.get("final_loss", r.get("loss"))
            if v is not None:
                return float(v)
    if log_name:
        lp = os.path.join(WORK, "logs", log_name)
        if os.path.exists(lp):
            last = None
            for line in open(lp):
                m = re.search(r"loss\s+([\d.]+)", line)
                if m:
                    last = float(m.group(1))
            if last is not None:
                return last
    return None

T0 = time.time()
SOFT_DEADLINE = T0 + GOVERNOR_HOURS * 3600
HARD_DEADLINE = SOFT_DEADLINE + HARD_MARGIN_MIN * 60


def log(msg):
    print(f"[patha {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- state

def state_path():
    return os.path.join(WORK, "state.json")


def load_state():
    if os.path.exists(state_path()):
        with open(state_path()) as f:
            return json.load(f)
    return {"stages": {}, "measured": {}, "steps_done": {}}


def save_state(st):
    os.makedirs(WORK, exist_ok=True)
    with open(state_path(), "w") as f:
        json.dump(st, f, indent=2)


def find_data(name):
    for root in (DATA_IN, WORK_DATA):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"{name} not found in {DATA_IN} or {WORK_DATA} -- attach the "
        f"opera-lm-patha-data dataset as a notebook input, or run "
        f"`python kaggle/patha_session.py --stages data` first")


def have_data(*names):
    try:
        for n in names:
            find_data(n)
        return True
    except FileNotFoundError:
        return False


def restore_ckpts():
    """Pull last session's runs + state from the attached ckpt dataset."""
    if not os.path.isdir(CKPT_IN):
        return
    for sub in ("runs", "artifacts"):
        src = os.path.join(CKPT_IN, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(WORK, sub), dirs_exist_ok=True)
    sp = os.path.join(CKPT_IN, "state.json")
    if os.path.exists(sp) and not os.path.exists(state_path()):
        shutil.copy(sp, state_path())
    log(f"restored runs/artifacts/state from {CKPT_IN}")


# ---------------------------------------------------------------- run

def _spawn(cmd, logfile, env_extra):
    """Start one job; returns (proc, logfile_handle, logfile_path) --
    the 3-tuple run_jobs/_watch unpack."""
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    f = open(logfile, "a")
    f.write(f"\n=== {time.ctime()} ===\n{' '.join(cmd)}\n")
    f.flush()
    full_env = dict(os.environ)
    full_env.update(env_extra)
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=f,
                            env=full_env, start_new_session=True)
    return proc, f, logfile


def _watch(procs):
    """Daemon thread: SIGTERM the whole process group at HARD_DEADLINE.
    Checkpoints every SAVE_EVERY steps make a hard stop safe."""
    def body():
        while any(p.poll() is None for p, _, _ in procs):
            if time.time() > HARD_DEADLINE:
                log(f"HARD DEADLINE: terminating {len(procs)} job(s)")
                for p, _, _ in procs:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                return
            time.sleep(20)
    threading.Thread(target=body, daemon=True).start()


def run_jobs(jobs):
    """jobs: list of (cmd, logfile, env_extra). Runs them concurrently
    under one watchdog, waits for all, prints tails of failures.
    Returns the list of return codes."""
    procs = []
    for cmd, logfile, env_extra in jobs:
        log(f"run: {' '.join(cmd[:8])}... -> {logfile}")
        procs.append(_spawn(cmd, os.path.join(WORK, "logs", logfile),
                            env_extra))
    _watch(procs)
    rcs = []
    for p, f, logfile in procs:
        rc = p.wait()
        f.close()
        if rc != 0:
            log(f"FAILED (rc={rc}); tail of {logfile}:")
            with open(os.path.join(WORK, "logs", logfile)) as tf:
                print("".join(tf.readlines()[-25:]), flush=True)
        rcs.append(rc)
    return rcs


def run(cmd, log_name, env=None):
    return run_jobs([(cmd, log_name, env or {})])[0]


def parse_s_per_step(logfile):
    """Last '<x>s/step' on a line that also shows 'T_cur 512' -- the
    steady-state number the session budgeting uses. (The log format is
    '(<cumulative>s, <avg>s/step)  T_cur 512': one paren wraps both
    numbers, so match the inner value, not a paren-prefixed one.)"""
    best = None
    with open(os.path.join(WORK, "logs", logfile)) as f:
        for line in f:
            if f"T_cur {MAX_LEN}" not in line:
                continue
            m = re.search(r"([\d.]+)s/step", line)
            if m:
                best = float(m.group(1))
    return best


def child_peak_rss_gb():
    """Peak RSS of children so far (Linux: KB; macOS: bytes -- Kaggle
    is Linux, the /2**20 reading is the KB case)."""
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20


# ---------------------------------------------------------------- push

def push_dataset(dirpath, slug, title, msg):
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        log("no kaggle credentials in env; skipping dataset push")
        return False
    meta_path = os.path.join(dirpath, "dataset-metadata.json")
    if not os.path.exists(meta_path):
        subprocess.run(["kaggle", "datasets", "init", "-p", dirpath],
                       check=True, capture_output=True)
        with open(meta_path) as f:
            meta = json.load(f)
        meta["title"], meta["id"] = title, slug
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        rc = subprocess.run(["kaggle", "datasets", "create", "-p", dirpath,
                             "--dir-mode", "zip"]).returncode
    else:
        rc = subprocess.run(["kaggle", "datasets", "version", "-p", dirpath,
                             "-m", msg, "--dir-mode", "zip"]).returncode
    log(f"dataset push ({slug}): rc={rc}")
    return rc == 0


def push_all(st, msg):
    user = os.environ.get("KAGGLE_USERNAME", "USER")
    push_dataset(WORK, f"{user}/opera-lm-patha-ckpt",
                 "opera-lm-patha-ckpt", msg)
    if os.path.isdir(WORK_DATA) and os.listdir(WORK_DATA):
        push_dataset(WORK_DATA, f"{user}/opera-lm-patha-data",
                     "opera-lm-patha-data", msg)


# ---------------------------------------------------------------- helpers

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def torch_ckpt_step(path):
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    return int(st.get("step", -1)) + 1


def steps_done(st, arm, phase):
    """Highest optimizer step reached by an arm's run so far."""
    out = os.path.join(WORK, "runs", f"{phase}_{arm}")
    if arm == "opera":
        cks = glob.glob(os.path.join(out, "*_train_ckpt.pt"))
        if cks:
            return torch_ckpt_step(max(cks, key=os.path.getmtime))
    else:
        ck = os.path.join(out, f"tf_{arm}_opt-muon_train_ckpt.pt")
        if os.path.exists(ck):
            return torch_ckpt_step(ck)
    return int(st["steps_done"].get(f"{phase}_{arm}", 0))


def opera_final(out_dir):
    finals = [f for f in glob.glob(os.path.join(out_dir, "*.pt"))
              if "_train_ckpt" not in f]
    return max(finals, key=os.path.getmtime) if finals else None


def next_log_idx(prefix):
    return len(glob.glob(os.path.join(WORK, "logs", f"{prefix}_*.log")))


# ---------------------------------------------------------------- stages

def stage_env(st):
    subprocess.run([PY, "-m", "pip", "install", "-q", "datasets",
                    "tokenizers", "huggingface_hub", "kaggle"], check=False)
    if not os.environ.get("SKIP_SELFTEST"):
        rc = run([PY, "-m", "opera_lm.selftest"], "env_selftest.log")
        assert rc == 0, "opera_lm selftest FAILED -- do not train on this"
    if _cuda_available():
        rc = run([PY, "-m", "opera_lm.triton_kernel", "--test"],
                 "env_triton_test.log")
        st["measured"]["triton_test_rc"] = rc
        log(f"triton kernel --test rc={rc} (0 = available; training does "
            f"not depend on it)")
    st["stages"]["env"] = "done"
    save_state(st)


def stage_data(st):
    """One-time data build (CPU session, no GPU quota). Writes
    WORK_DATA and pushes after EVERY artifact -- a mid-stage session
    kill (cap hit, accelerator switch, manual stop) then loses at most
    the one artifact in flight, not the whole build, because the pushed
    dataset is what the next session re-attaches ( Kaggle wipes
    /kaggle/working on every session restart)."""
    os.makedirs(WORK_DATA, exist_ok=True)

    # (1) 2M-line tokenizer -- the Mac-segfault fix is cloud RAM; staged
    # fallback 2M -> 1.5M -> 1M per prereg §5.1's spirit (measured).
    if not have_data("tokenizer_2m.json"):
        tok_2m = os.path.join(WORK_DATA, "tokenizer_2m.json")
        for docs in (2_000_000, 1_500_000, 1_000_000):
            rc = run([PY, "opera-chat/train_tokenizer.py",
                      "--docs", str(docs), "--out", tok_2m],
                     f"data_tokenizer_{docs}.log")
            peak = child_peak_rss_gb()
            log(f"tokenizer {docs} lines: rc={rc}, child peak RSS "
                f"{peak:.1f} GB")
            if rc == 0:
                st["measured"]["tokenizer_docs"] = docs
                st["measured"]["tokenizer_peak_rss_gb"] = round(peak, 1)
                break
        assert have_data("tokenizer_2m.json"), \
            "tokenizer training failed at 1M+"
    save_state(st)
    push_all(st, "data: tokenizer built")

    # (2) saturation check + adoption rule (prereg §5.1). Judged by the
    # ARTIFACT, not the exit code: HF-streaming scripts can abort during
    # interpreter finalization AFTER writing their output (the repo's
    # known downloader-thread landmine; saturation_check now os._exit(0)s
    # too, but a half-dead rc with a complete json is a pass).
    if not have_data("tokenizer_saturation.json"):
        run([PY, os.path.join(REPO, "kaggle", "saturation_check.py"),
             "--old", os.path.join(REPO, "opera-chat", "tokenizer.json"),
             "--new", find_data("tokenizer_2m.json"),
             "--out", os.path.join(WORK_DATA,
                                   "tokenizer_saturation.json")],
            "data_saturation.log")
        assert have_data("tokenizer_saturation.json"), \
            "saturation check failed (no output json)"
    with open(find_data("tokenizer_saturation.json")) as f:
        sat = json.load(f)
    st["measured"]["tokenizer_saturation"] = sat
    chosen = (find_data("tokenizer_2m.json") if sat.get("adopt_new")
              else os.path.join(REPO, "opera-chat", "tokenizer.json"))
    shutil.copy(chosen, os.path.join(WORK_DATA, "tokenizer.json"))
    log(f"tokenizer adopted: {'NEW 2M-line' if sat.get('adopt_new') else 'OLD (§4.8 comparability)'}")
    save_state(st)
    push_all(st, "data: tokenizer adopted")

    # (3) smoltalk pools. have_data() checks the pushed dataset too, so
    # a session resuming after a mid-stage kill skips whatever already
    # reached the dataset and rebuilds only what did not.
    if not have_data("data_smoltalk_512.pkl"):
        rc = run([PY, "opera-chat/prepare_data.py",
                  "--tokenizer", find_data("tokenizer.json"),
                  "--out", os.path.join(WORK_DATA, "data_smoltalk_512.pkl"),
                  "--n-convs", "5000000", "--max-len", str(MAX_LEN),
                  "--eval-max-len", str(EVAL_MAX)],
                 "data_prepare_smoltalk.log")
        assert rc == 0, "smoltalk prep failed"
        push_all(st, "data: smoltalk pool built")

    # (4) FineWeb-Edu pretrain pool. prepare_fineweb's --max-tokens caps
    # STREAMED tokens; doc_chunks(long_first=True) routes only ~9% of
    # them into <=512 train chunks (measured 2026-09-05: 250M streamed
    # -> 22.1M-token pool; §4.8 had the same shape, 500M -> 19.9M). So
    # the pool is judged by its own packed size against
    # PATHA_FINWEB_POOL_MIN (default 80M), streaming up to
    # PATHA_FINWEB_TOKENS (default 1B) to get there. A session with an
    # outdated smaller pool attached rebuilds + pushes; the NEXT
    # session's re-attached dataset then passes the check.
    fw_name = "data_fineweb_512"
    fin_stream = int(os.environ.get("PATHA_FINWEB_TOKENS", "1000000000"))
    fin_pool_min = int(os.environ.get("PATHA_FINWEB_POOL_MIN", "80000000"))
    pool_stats = None
    if have_data(fw_name + ".pack.json"):
        with open(find_data(fw_name + ".pack.json")) as f:
            pool_stats = json.load(f)
        log(f"fineweb pool on hand: {pool_stats['total_tokens']:,} "
            f"unique train tokens (min {fin_pool_min:,})")
    rebuilt_fw = False
    if not (pool_stats and pool_stats["total_tokens"] >= fin_pool_min):
        rc = run([PY, "opera-chat/prepare_fineweb.py",
                  "--tokenizer", find_data("tokenizer.json"),
                  "--out", os.path.join(WORK_DATA, fw_name + ".pkl"),
                  "--max-tokens", str(fin_stream),
                  "--max-len", str(MAX_LEN),
                  "--eval-max-len", str(EVAL_MAX)],
                 "data_prepare_fineweb.log")
        assert rc == 0, "fineweb prep failed"
        rebuilt_fw = True
        push_all(st, "data: fineweb pool rebuilt (bigger stream)")

    # (5) pack both train pools -> eval-only pkl + npy pair. The source
    # pkl may sit on the read-only dataset mount (read is fine); pack
    # OUTPUTS always go to the writable WORK_DATA, then push. A freshly
    # rebuilt fineweb pool MUST repack from the local pkl (the mounted
    # one is the stale small version), hence force=rebuilt_fw.
    for name, force in (("data_smoltalk_512", False), (fw_name, rebuilt_fw)):
        prefix = os.path.join(WORK_DATA, name)
        if force or not have_data(name + ".tokens.npy"):
            pkl_src = (os.path.join(WORK_DATA, name + ".pkl") if force
                       else find_data(name + ".pkl"))
            rc = run([PY, os.path.join(REPO, "kaggle", "pack_data.py"),
                      "--pkl", pkl_src, "--prefix", prefix,
                      "--max-len", str(MAX_LEN)], "data_pack.log")
            assert rc == 0, f"packing {name} failed"
            push_all(st, f"data: {name} packed")

    # (6) bucket-population report (prereg §5.2)
    run([PY, os.path.join(REPO, "kaggle", "bucket_report.py"),
         "--smoltalk", find_data("data_smoltalk_512.eval.pkl"),
         "--fineweb", find_data("data_fineweb_512.eval.pkl"),
         "--max-len", str(MAX_LEN), "--eval-max-len", str(EVAL_MAX),
         "--out", os.path.join(WORK_DATA, "bucket_report.json")],
        "data_buckets.log")

    st["stages"]["data"] = "done"
    save_state(st)
    push_all(st, "data stage complete")


def stage_smoke(st):
    assert have_data("data_smoltalk_512.eval.pkl"), "run the data stage first"
    smol_pkl = find_data("data_smoltalk_512.eval.pkl")
    smol_prefix = smol_pkl[:-len(".eval.pkl")]

    # (a) 20M / 500 steps end-to-end: packed source + 8k eval + curve.
    # Fast curriculum (every 50) so ~350 steps run at T_cur 512.
    out20 = os.path.join(WORK, "runs", "smoke20m")
    if not glob.glob(os.path.join(out20, "*.pt")):
        base = [PY, "opera-chat/train_chat.py", "--data", smol_pkl,
                "--packed-data", smol_prefix, "--steps", "500",
                "--batch", "16", "--curriculum-every", "50",
                *SMOKE_OPERA_ARGS, "--max-len", str(MAX_LEN),
                "--eval-max-len", str(EVAL_MAX), "--device", "cuda",
                "--optimizer", "muon", "--muon-lr", "0.02",
                "--save-every", "0", "--out-dir", out20]
        rc = run(base + ["--compile", "default"], "smoke_20m.log")
        if rc != 0:   # dynamo trouble on this stack: eager fallback (the
            # existing Kaggle notebook's documented pattern)
            rc = run(base + ["--compile", "off"], "smoke_20m_eager.log")
            assert rc == 0, "20M smoke failed even in eager mode"
    st["stages"]["smoke20m"] = "done"
    save_state(st)

    # (b) position_curve on the smoke checkpoint (8k instrument check)
    ckpt = opera_final(out20)
    assert ckpt, "smoke produced no final checkpoint"
    os.makedirs(os.path.join(WORK, "artifacts"), exist_ok=True)
    rc = run([PY, "opera-chat/position_curve.py", "--ckpt", ckpt,
              "--config", os.path.join(out20, "model_config.json"),
              "--data", smol_pkl, "--device", "cuda",
              "--train-len", str(MAX_LEN), "--max-len", str(EVAL_MAX),
              "--n", "50", "--batch", "2",
              "--out", os.path.join(WORK, "artifacts", "pc_smoke20m.json")],
             "smoke_position_curve.log")
    assert rc == 0, "position_curve failed on the smoke checkpoint"
    st["stages"]["smoke_curve"] = "done"
    save_state(st)

    # (c) 155M throughput measurements (budget sizing for the sessions)
    fw_pkl = find_data("data_fineweb_512.eval.pkl")
    fw_prefix = fw_pkl[:-len(".eval.pkl")]
    if "s_per_step_opera" not in st["measured"]:
        rc = run(["torchrun", "--nproc_per_node=2",
                  "opera-chat/train_chat.py", "--ddp",
                  "--data", fw_pkl, "--packed-data", fw_prefix,
                  "--steps", "40", "--batch", "16",
                  "--curriculum-every", "10", *OPERA_ARGS,
                  "--max-len", str(MAX_LEN), "--eval-max-len", str(EVAL_MAX),
                  "--device", "cuda", "--optimizer", "muon",
                  "--muon-lr", "0.02", "--compile", "off",
                  "--grad-checkpoint", "level", "--save-every", "0",
                  "--out-dir", os.path.join(WORK, "runs", "smoke_tp_opera")],
                 "smoke_tp_opera.log")
        s = parse_s_per_step("smoke_tp_opera.log")
        assert s, "could not parse OPERA s/step from throughput smoke"
        st["measured"]["s_per_step_opera"] = s
        log(f"MEASURED: OPERA 155M 2xT4 DDP (eff. batch 32) = {s:.2f} s/step")
    if "s_per_step_tf" not in st["measured"]:
        run([PY, "opera-chat/train_tf_chat.py", "--pe", "rope",
             "--data", fw_pkl, "--packed-data", fw_prefix,
             "--steps", "60", "--batch", "32", "--curriculum-every", "10",
             *TF_ARGS, "--device", "cuda", "--optimizer", "muon",
             "--muon-lr", "0.02", "--lr-schedule", "wsd", "--save-every", "0",
             "--out-dir", os.path.join(WORK, "runs", "smoke_tp_tf")],
            "smoke_tp_tf.log")
        s = parse_s_per_step("smoke_tp_tf.log")
        assert s, "could not parse TF s/step from throughput smoke"
        st["measured"]["s_per_step_tf"] = s
        log(f"MEASURED: TF 155M 1xT4 batch 32 = {s:.2f} s/step")
    # (d) RECIPE SWEEP (prereg amendment 2026-09-09) + 155M confirm:
    # the recipe is measured on this study's own data and hardware
    # before any long training runs; adoption rules are pre-stated in
    # _recipe_sweep's docstring and the prereg amendment.
    if "recipe" not in st:
        _recipe_sweep(st, fw_pkl, fw_prefix)
    st["stages"]["recipe"] = "done"
    save_state(st)
    st["stages"]["smoke"] = "done"
    save_state(st)
    push_all(st, "smoke stage complete (recipe measured + adopted)")



def _recipe_probe_cmd(kind, name, lr, wd, include, pkl, prefix, out_dir,
                      tf=False):
    if tf:
        return [PY, "opera-chat/train_tf_chat.py", "--pe", "rope",
                "--data", pkl, "--packed-data", prefix,
                "--steps", str(SWEEP_STEPS), "--batch", "32",
                *SMOKE_TF_ARGS, "--device", "cuda", "--optimizer", "muon",
                "--muon-lr", str(lr), "--muon-wd", str(wd),
                "--lr-schedule", "wsd", "--curriculum-every", "50",
                "--save-every", "0", "--out-dir", out_dir]
    cmd = [PY, "opera-chat/train_chat.py", "--data", pkl,
           "--packed-data", prefix, "--steps", str(SWEEP_STEPS),
           "--batch", "16", "--curriculum-every", "50",
           *SMOKE_OPERA_ARGS, "--max-len", str(MAX_LEN),
           "--eval-max-len", str(EVAL_MAX), "--device", "cuda",
           "--optimizer", "muon", "--muon-lr", str(lr),
           "--muon-wd", str(wd), "--compile", "off",
           "--save-every", "0", "--out-dir", out_dir]
    if include:
        cmd += ["--muon-include", include]
    return cmd


def _recipe_sweep(st, pkl, prefix):
    """Session-2 recipe sweep (prereg amendment 2026-09-09): measure the
    optimizer recipe ON THIS STUDY'S data/tokenization/hardware at the
    20M rung before the 3-week commitment, symmetric for both arms.
    Pre-stated adoption rules:
      lr   argmin over {0.01, 0.02, 0.04} (wd 0.01, fgate on); keep 0.02
           unless the winner beats it by > 1%.
      wd   0 vs 0.01 at the winning lr; keep 0.01 unless 0 wins by > 1%.
      fg   fusion_gate stays (correctness fix) unless off wins by > 1.5%.
      tf   same lr rule at wd 0.01 (TF has no fusion analogue).
      155M top-2 confirmation: an at-scale ranking flip OVERRIDES the
      20M lr ranking (pre-stated).
    """
    root = os.path.join(WORK, "runs", "recipe")
    def _k(pref, lr):
        return f"{pref}_lr{int(lr * 1000):03d}"
    opera_cfgs = [(_k("op", 0.01), 0.01, 0.01, True),
                  (_k("op", 0.02), 0.02, 0.01, True),
                  (_k("op", 0.04), 0.04, 0.01, True),
                  ("op_wd0", 0.02, 0.0, True),
                  ("op_nofg", 0.02, 0.01, False)]
    tf_cfgs = [(_k("tf", 0.01), 0.01), (_k("tf", 0.02), 0.02),
               (_k("tf", 0.04), 0.04)]
    todo = [(n, _recipe_probe_cmd("opera", n, lr, wd, "fusion_gate" if fg else "",
                                  pkl, prefix, os.path.join(root, n)),
             "0" if i % 2 == 0 else "1")
            for i, (n, lr, wd, fg) in enumerate(opera_cfgs)]
    todo += [(n, _recipe_probe_cmd("tf", n, lr, 0.01, "",
                                   pkl, prefix, os.path.join(root, n)),
              "0" if i % 2 == 0 else "1")
             for i, (n, lr) in enumerate(tf_cfgs)]
    losses = {}
    for k in range(0, len(todo), 2):
        batch = todo[k:k + 2]
        run_jobs([(cmd, f"recipe_{n}.log", {"CUDA_VISIBLE_DEVICES": dev})
                  for n, cmd, dev in batch])
    for n, cmd, dev in todo:
        v = _final_loss(os.path.join(root, n), f"recipe_{n}.log")
        assert v is not None, f"probe {n} produced no final loss"
        losses[n] = v
        log(f"SWEEP {n}: final loss {v:.4f}")
    # adoption (rules above, arithmetic not judgement)
    lr_best = min(("0.01", "0.02", "0.04"), key=lambda l: losses[_k("op", l)])
    keep = ("0.02" if losses[_k("op", "0.02")] <= losses[_k("op", lr_best)] * 1.01
            else lr_best)
    wd = "0.01" if losses["op_wd0"] > losses[_k("op", "0.02")] * 0.99 else "0.0"
    inc = ("fusion_gate"
           if losses["op_nofg"] > losses[_k("op", "0.02")] * 0.985 else "")
    tf_best = min(("0.01", "0.02", "0.04"), key=lambda l: losses[_k("tf", l)])
    tf_keep = ("0.02" if losses[_k("tf", "0.02")] <= losses[_k("tf", tf_best)] * 1.01
               else tf_best)
    # 155M confirmation of the lr decision (at-scale overrides 20M)
    c_a, c_b = keep, "0.02"
    if c_a != c_b:
        outs = {}
        for tag, lr_v, dev in ((f"confirm_{c_a}", c_a, "0"), (f"confirm_{c_b}", c_b, "1")):
            run(["torchrun", "--nproc_per_node=2", "opera-chat/train_chat.py",
                 "--ddp", "--data", pkl, "--packed-data", prefix,
                 "--steps", str(PROBE_STEPS), "--batch", "16", *OPERA_ARGS,
                 "--max-len", str(MAX_LEN), "--eval-max-len", str(EVAL_MAX),
                 "--device", "cuda", "--optimizer", "muon",
                 "--muon-lr", lr_v, "--muon-include", "fusion_gate",
                 "--muon-wd", "0.01", "--curriculum-every", "10",
                 "--compile", "off", "--grad-checkpoint", "level",
                 "--save-every", "0",
                 "--out-dir", os.path.join(root, tag)],
                f"recipe_{tag}.log")
            outs[lr_v] = _final_loss(os.path.join(root, tag),
                                     f"recipe_{tag}.log")
        keep = min(outs, key=outs.get)
        log(f"155M confirm: {outs} -> lr {keep} adopted")
    st["recipe"] = {"opera": {"lr": keep, "wd": wd, "include": inc},
                    "tf": {"lr": tf_keep, "wd": "0.01"},
                    "evidence": losses}
    log(f"RECIPE ADOPTED: opera lr={keep} wd={wd} include={inc or 'none'} | "
        f"tf lr={tf_keep} wd=0.01")
    save_state(st)
    push_all(st, "recipe sweep complete")


def stage_pretrain_opera(st):
    fw_pkl = find_data("data_fineweb_512.eval.pkl")
    fw_prefix = fw_pkl[:-len(".eval.pkl")]
    done = steps_done(st, "opera", "pretrain")
    s = st["measured"]["s_per_step_opera"]
    fit = int(max(0.0, SOFT_DEADLINE - time.time() - 600) / s)
    log(f"OPERA pretrain: {done}/{PRETRAIN_STEPS} steps done; ~{fit} more "
        f"fit before the governor at {s:.2f} s/step")
    run(["torchrun", "--nproc_per_node=2", "opera-chat/train_chat.py",
         "--ddp", "--data", fw_pkl, "--packed-data", fw_prefix,
         "--steps", str(PRETRAIN_STEPS), "--batch", "16", *OPERA_ARGS,
         "--max-len", str(MAX_LEN), "--eval-max-len", str(EVAL_MAX),
         "--device", "cuda", "--optimizer", "muon",
         "--muon-lr", rcp["opera"]["lr"], "--muon-include",
         rcp["opera"]["include"], "--muon-wd", rcp["opera"]["wd"],
         "--lr-schedule", "wsd", "--save-every", str(SAVE_EVERY),
         "--resume", "--compile", "off", "--grad-checkpoint", "level",
         "--out-dir", os.path.join(WORK, "runs", "pretrain_opera")],
        f"pretrain_opera_{next_log_idx('pretrain_opera')}.log")
    st["steps_done"]["pretrain_opera"] = steps_done(st, "opera", "pretrain")
    if (st["steps_done"]["pretrain_opera"] >= PRETRAIN_STEPS
            and opera_final(os.path.join(WORK, "runs", "pretrain_opera"))):
        st["stages"]["pretrain_opera"] = "done"
    save_state(st)
    push_all(st, "pretrain opera progress")


def _tf_train_cmd(pe, pkl, prefix, steps, out_dir, init=None, rcp=None):
    rcp = rcp or RECIPE_DEFAULT
    cmd = [PY, "opera-chat/train_tf_chat.py", "--pe", pe, "--data", pkl,
           "--packed-data", prefix, "--steps", str(steps), "--batch", "32",
           *TF_ARGS, "--device", "cuda", "--optimizer", "muon",
           "--muon-lr", rcp["tf"]["lr"], "--muon-wd", rcp["tf"]["wd"],
           "--lr-schedule", "wsd",
           "--save-every", str(SAVE_EVERY), "--resume",
           "--out-dir", out_dir]
    if init:
        cmd += ["--init-weights-from", init]
    return cmd


def stage_pretrain_tf(st):
    """rope + nope pretrains concurrently, one per T4."""
    fw_pkl = find_data("data_fineweb_512.eval.pkl")
    fw_prefix = fw_pkl[:-len(".eval.pkl")]
    n = next_log_idx("pretrain_tf")
    jobs = []
    for pe in ("rope", "nope"):
        if steps_done(st, pe, "pretrain") >= PRETRAIN_STEPS:
            continue
        jobs.append((_tf_train_cmd(pe, fw_pkl, fw_prefix, PRETRAIN_STEPS,
                                   os.path.join(WORK, "runs", f"pretrain_{pe}")),
                     f"pretrain_tf_{n}_{pe}.log",
                     {"CUDA_VISIBLE_DEVICES": "0" if pe == "rope" else "1"}))
    if jobs:
        run_jobs(jobs)
    for pe in ("rope", "nope"):
        st["steps_done"][f"pretrain_{pe}"] = steps_done(st, pe, "pretrain")
        if st["steps_done"][f"pretrain_{pe}"] >= PRETRAIN_STEPS:
            st["stages"][f"pretrain_{pe}"] = "done"
    save_state(st)
    push_all(st, "pretrain tf pair progress")


def stage_sft_opera(st):
    smol_pkl = find_data("data_smoltalk_512.eval.pkl")
    smol_prefix = smol_pkl[:-len(".eval.pkl")]
    init = opera_final(os.path.join(WORK, "runs", "pretrain_opera"))
    assert init, "opera pretrain has no final checkpoint yet"
    rcp = get_recipe(st)
    run(["torchrun", "--nproc_per_node=2", "opera-chat/train_chat.py",
         "--ddp", "--data", smol_pkl, "--packed-data", smol_prefix,
         "--steps", str(SFT_STEPS), "--batch", "16", *OPERA_ARGS,
         "--max-len", str(MAX_LEN), "--eval-max-len", str(EVAL_MAX),
         "--device", "cuda", "--optimizer", "muon",
         "--muon-lr", rcp["opera"]["lr"], "--muon-include",
         rcp["opera"]["include"], "--muon-wd", rcp["opera"]["wd"],
         "--lr-schedule", "wsd", "--save-every", str(SAVE_EVERY),
         "--resume", "--compile", "off", "--grad-checkpoint", "level",
         "--init-weights-from", init,
         "--out-dir", os.path.join(WORK, "runs", "sft_opera")],
        f"sft_opera_{next_log_idx('sft_opera')}.log")
    st["steps_done"]["sft_opera"] = steps_done(st, "opera", "sft")
    if (st["steps_done"]["sft_opera"] >= SFT_STEPS
            and opera_final(os.path.join(WORK, "runs", "sft_opera"))):
        st["stages"]["sft_opera"] = "done"
    save_state(st)
    push_all(st, "sft opera progress")


def stage_sft_tf(st):
    """rope + nope SFT concurrently (or whichever remains)."""
    smol_pkl = find_data("data_smoltalk_512.eval.pkl")
    smol_prefix = smol_pkl[:-len(".eval.pkl")]
    n = next_log_idx("sft_tf")
    jobs = []
    for pe in ("rope", "nope"):
        if st["stages"].get(f"sft_{pe}") == "done":
            continue
        init = os.path.join(WORK, "runs", f"pretrain_{pe}",
                            f"tf_{pe}_opt-muon.pt")
        assert os.path.exists(init), f"{init} missing (pretrain unfinished)"
        jobs.append((_tf_train_cmd(pe, smol_pkl, smol_prefix, SFT_STEPS,
                                   os.path.join(WORK, "runs", f"sft_{pe}"),
                                   init=init),
                     f"sft_tf_{n}_{pe}.log",
                     {"CUDA_VISIBLE_DEVICES": "0" if pe == "rope" else "1"}))
    if jobs:
        run_jobs(jobs)
    for pe in ("rope", "nope"):
        if os.path.exists(os.path.join(WORK, "runs", f"sft_{pe}",
                                       f"tf_{pe}_opt-muon.pt")):
            st["stages"][f"sft_{pe}"] = "done"
    save_state(st)
    push_all(st, "sft tf pair progress")


def stage_curves(st):
    smol_pkl = find_data("data_smoltalk_512.eval.pkl")
    os.makedirs(os.path.join(WORK, "artifacts"), exist_ok=True)
    curve_jobs = []
    o_final = opera_final(os.path.join(WORK, "runs", "sft_opera"))
    assert o_final, "sft_opera final checkpoint missing"
    if not os.path.exists(os.path.join(WORK, "artifacts", "pc_opera.json")):
        curve_jobs.append(([PY, "opera-chat/position_curve.py",
                            "--ckpt", o_final, "--config",
                            os.path.join(WORK, "runs", "sft_opera",
                                         "model_config.json"),
                            "--data", smol_pkl, "--device", "cuda",
                            "--train-len", str(MAX_LEN),
                            "--max-len", str(EVAL_MAX),
                            "--out", os.path.join(WORK, "artifacts",
                                                  "pc_opera.json")],
                           "curves_opera.log", {}))
    for pe in ("rope", "nope"):
        outj = os.path.join(WORK, "artifacts", f"pc_{pe}.json")
        if os.path.exists(outj):
            continue
        curve_jobs.append(([PY, "opera-chat/position_curve.py", "--tf",
                            "--pe", pe, *TF_ARGS,
                            "--ckpt", os.path.join(WORK, "runs", f"sft_{pe}",
                                                   f"tf_{pe}_opt-muon.pt"),
                            "--data", smol_pkl, "--device", "cuda",
                            "--train-len", str(MAX_LEN),
                            "--max-len", str(EVAL_MAX), "--out", outj],
                           f"curves_{pe}.log",
                           {"CUDA_VISIBLE_DEVICES": "0" if pe == "rope" else "1"}))
    rcs = run_jobs(curve_jobs)
    assert all(rc == 0 for rc in rcs), "a position curve failed"
    rc = run([PY, os.path.join(REPO, "kaggle", "make_artifacts.py"),
              "--work", WORK, "--data", smol_pkl,
              "--tokenizer", find_data("tokenizer.json")],
             "curves_artifacts.log")
    assert rc == 0, "artifact assembly failed"
    st["stages"]["curves"] = "done"
    save_state(st)
    push_all(st, "curves + artifacts complete")


# ---------------------------------------------------------------- auto

def auto(st):
    if "env" not in st["stages"]:
        stage_env(st)
    if not have_data("data_smoltalk_512.eval.pkl"):
        stage_data(st)
    if not _cuda_available():
        log("no GPU in this session (CPU data session) -- stopping here")
        push_all(st, "cpu session end")
        return
    if st["stages"].get("smoke") != "done":
        stage_smoke(st)
    if steps_done(st, "opera", "pretrain") < PRETRAIN_STEPS:
        stage_pretrain_opera(st)
        return
    if (steps_done(st, "rope", "pretrain") < PRETRAIN_STEPS
            or steps_done(st, "nope", "pretrain") < PRETRAIN_STEPS):
        stage_pretrain_tf(st)
        return
    if st["stages"].get("sft_opera") != "done":
        stage_sft_opera(st)
        return
    if (st["stages"].get("sft_rope") != "done"
            or st["stages"].get("sft_nope") != "done"):
        stage_sft_tf(st)
        return
    if st["stages"].get("curves") != "done":
        stage_curves(st)
    push_all(st, "session end")
    log("STUDY COMPLETE" if st["stages"].get("curves") == "done"
        else "session stopped -- rerun the notebook to continue")


def status(st):
    print(json.dumps(st, indent=2))
    print(f"data attached: "
          f"{have_data('data_smoltalk_512.eval.pkl', 'data_fineweb_512.eval.pkl')}")
    for arm in ("opera", "rope", "nope"):
        for phase in ("pretrain", "sft"):
            print(f"{phase}_{arm}: {steps_done(st, arm, phase)} steps "
                  f"(target {PRETRAIN_STEPS if phase == 'pretrain' else SFT_STEPS})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="auto",
                   help="comma list from {auto,env,data,smoke,pretrain_opera,"
                        "pretrain_tf,sft_opera,sft_tf,curves,status}")
    a = p.parse_args()
    os.makedirs(WORK, exist_ok=True)
    restore_ckpts()
    st = load_state()
    for sname in a.stages.split(","):
        if time.time() > SOFT_DEADLINE:
            log("past soft deadline; pushing and stopping")
            break
        {"auto": lambda: auto(st),
         "status": lambda: status(st),
         "env": lambda: stage_env(st),
         "data": lambda: stage_data(st),
         "smoke": lambda: stage_smoke(st),
         "pretrain_opera": lambda: stage_pretrain_opera(st),
         "pretrain_tf": lambda: stage_pretrain_tf(st),
         "sft_opera": lambda: stage_sft_opera(st),
         "sft_tf": lambda: stage_sft_tf(st),
         "curves": lambda: stage_curves(st)}[sname]()
    push_all(st, "session end")


if __name__ == "__main__":
    main()
