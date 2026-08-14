"""chat_common.py -- shared chat token format for the opera-chat pipeline.

Special tokens (fixed ids; the tokenizer is trained with these first, so
the ids below are canonical -- <|pad|> MUST be 0 because opera_lm zero-pads
batches):

    0 <|pad|>   1 <|user|>   2 <|assistant|>   3 <|end|>   4 <|system|>

Per-conversation layout: for each message in order,
    <role token> + BPE(text.strip())
and assistant messages additionally get a trailing <|end|>. A generation
prompt is the formatted history followed by a bare <|assistant|> token;
the stop token for generation is <|end|>. <|pad|> is never emitted.
"""
import os
import sys

SPECIAL_TOKENS = ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>",
                  "<|system|>"]

PAD_ID = 0
USER_ID = 1
ASSISTANT_ID = 2
END_ID = 3
SYSTEM_ID = 4

# role name -> special token name
ROLE_TOKEN = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
END_TOKEN = "<|end|>"


def ensure_opera_lm():
    """Make `import opera_lm` work without installing the package: put
    the repo root (parent of this file's opera-chat/ dir, which directly
    contains the opera_lm/ package) on sys.path."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def encode_ids(tok, text):
    """tokenizers.Tokenizer.encode returns an Encoding; the selftest's
    DummyTokenizer returns a plain list. Accept both."""
    enc = tok.encode(text)
    return list(enc.ids) if hasattr(enc, "ids") else list(enc)


def format_conversation(messages, tok):
    """messages: list of {"role", "content"} dicts (roles system/user/
    assistant). Returns the flat token-id list for the conversation."""
    ids = []
    for m in messages:
        ids.append(tok.token_to_id(ROLE_TOKEN[m["role"]]))
        ids.extend(encode_ids(tok, m["content"].strip()))
        if m["role"] == "assistant":
            ids.append(tok.token_to_id(END_TOKEN))
    return ids


def build_prompt(history, tok):
    """Formatted history plus the trailing <|assistant|> token that opens
    the reply the model should generate."""
    return format_conversation(history, tok) + \
        [tok.token_to_id(ROLE_TOKEN["assistant"])]


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)
