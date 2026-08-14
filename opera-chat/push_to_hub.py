"""push_to_hub.py -- deploy the chat model + Gradio Space to Hugging Face.

Creates/updates two repos:
  - model repo: checkpoint, model_config.json, tokenizer.json
  - space repo: app.py, requirements.txt, README.md, and a self-contained
    copy of the opera_lm package (the Space does not pip-install it).

The uploaded app.py has its HF_REPO default patched to the model repo id
(the line marked `# HF_REPO_DEFAULT`), so the Space needs zero secrets or
configuration.

Usage:
  python push_to_hub.py --ckpt runs_chat/opera_v8_0_....pt \
      --config runs_chat/model_config.json \
      --model-repo USER/opera-lm-chat --space-repo USER/opera-lm-chat-space
Token: --token or the HF_TOKEN environment variable.
"""
import argparse
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

# the subset of opera_lm the Space needs at inference time
OPERA_LM_FILES = ["__init__.py", "model.py", "incremental.py", "losses.py",
                  "train.py", "data.py", "metal_kernel.py"]

HF_REPO_MARKER = "# HF_REPO_DEFAULT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", default="opera-chat/tokenizer.json")
    p.add_argument("--model-repo", required=True)
    p.add_argument("--space-repo", required=True)
    p.add_argument("--token", default=None)
    a = p.parse_args()

    token = a.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: no HF token -- pass --token or set HF_TOKEN")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    # --- model repo ---
    api.create_repo(a.model_repo, repo_type="model", exist_ok=True)
    for path in [a.ckpt, a.config, a.tokenizer]:
        api.upload_file(path_or_fileobj=path,
                        path_in_repo=os.path.basename(path),
                        repo_id=a.model_repo, repo_type="model")
        print(f"  uploaded {path} -> {a.model_repo}", flush=True)

    # --- space repo: assemble a fresh dir in a temp location ---
    api.create_repo(a.space_repo, repo_type="space", space_sdk="gradio",
                    exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        # app.py with the HF_REPO default patched to the model repo
        with open(os.path.join(HERE, "app.py")) as f:
            app_src = f.read()
        lines, patched = app_src.splitlines(), False
        for i, ln in enumerate(lines):
            if HF_REPO_MARKER in ln:
                lines[i] = (f'HF_REPO = os.environ.get("HF_REPO", '
                            f'"{a.model_repo}")  {HF_REPO_MARKER}')
                patched = True
        assert patched, f"app.py is missing the {HF_REPO_MARKER} line"
        with open(os.path.join(tmp, "app.py"), "w") as f:
            f.write("\n".join(lines) + "\n")

        shutil.copy(os.path.join(HERE, "generate_chat.py"), tmp)
        shutil.copy(os.path.join(HERE, "chat_common.py"), tmp)
        shutil.copy(os.path.join(HERE, "requirements.txt"), tmp)
        shutil.copy(os.path.join(HERE, "space_README.md"),
                    os.path.join(tmp, "README.md"))

        pkg = os.path.join(tmp, "opera_lm")
        os.makedirs(pkg)
        for name in OPERA_LM_FILES:
            shutil.copy(os.path.join(REPO_ROOT, "opera_lm", name), pkg)

        api.upload_folder(folder_path=tmp, repo_id=a.space_repo,
                          repo_type="space")
    print(f"model: https://huggingface.co/{a.model_repo}", flush=True)
    print(f"space: https://huggingface.co/spaces/{a.space_repo}",
          flush=True)


if __name__ == "__main__":
    main()
