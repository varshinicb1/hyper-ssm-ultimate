"""
Infinite Context Memory -- Interactive CLI Chat
O(1) conversation memory with beautiful terminal UI.

Usage:
    python applications/cli_chat.py
    python applications/cli_chat.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    python applications/cli_chat.py --embedding-model "all-MiniLM-L6-v2"
    python applications/cli_chat.py --load session.pkl
"""

import sys
import os
import time
import pickle
import atexit
import json
import textwrap
import argparse
from pathlib import Path
from typing import Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from hyper_ssm.conversation_memory import InfiniteContextLLM

try:
    import readline
except ImportError:
    try:
        import pyreadline3 as readline
    except ImportError:
        readline = None

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    import colorama
    colorama.init()
    _HAS_COLORAMA = True
except Exception:
    _HAS_COLORAMA = False

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------
if _HAS_COLORAMA:
    import colorama
    _CYAN = colorama.Fore.CYAN
    _GREEN = colorama.Fore.GREEN
    _YELLOW = colorama.Fore.YELLOW
    _MAGENTA = colorama.Fore.MAGENTA
    _RED = colorama.Fore.RED
    _WHITE = colorama.Fore.WHITE
    _BOLD = colorama.Style.BRIGHT
    _DIM = colorama.Style.DIM
    _RESET = colorama.Style.RESET_ALL
else:
    _CYAN = "\033[96m"
    _GREEN = "\033[92m"
    _YELLOW = "\033[93m"
    _MAGENTA = "\033[95m"
    _RED = "\033[91m"
    _WHITE = "\033[97m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"


def c(s: str, color: str) -> str:
    return f"{color}{s}{_RESET}"


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
BANNER = f"""
{_MAGENTA}{_BOLD}  ===== INFINITE CONTEXT MEMORY ====={_RESET}
{_CYAN}         O(1) Conversation State{_RESET}
"""

# ---------------------------------------------------------------------------
# Readline tab completer
# ---------------------------------------------------------------------------
COMMANDS = [
    "/help", "/info", "/save", "/load", "/reset", "/history",
    "/model", "/scale", "/stats", "/quit", "/exit",
]


class Completer:
    def __init__(self):
        self.matches: list[str] = []

    def complete(self, text: str, state: int) -> Optional[str]:
        if state == 0:
            if text.startswith("/"):
                self.matches = [c for c in COMMANDS if c.startswith(text)]
            else:
                self.matches = []
        try:
            return self.matches[state]
        except IndexError:
            return None


# ---------------------------------------------------------------------------
# Simulated LLM (when no real model available)
# ---------------------------------------------------------------------------
class SimulatedLLM:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.rng = np.random.RandomState(42)
        self.responses = [
            "That's an interesting point! From my infinite context memory, "
            "I recall we've been discussing this topic.",
            "Based on everything you've told me, I think the key insight here "
            "is about maintaining O(1) memory compression.",
            "Let me check my hyperbolic state... Yes, I remember you "
            "mentioned something similar earlier!",
            "Great question! The infinite context memory allows me to recall "
            "details from anywhere in our conversation.",
            "I've compressed that into my Lorentzian state vector. Here's "
            "what I recall about that topic.",
        ]

    def __call__(self, prompt: str) -> str:
        delay = 0.1 + 0.3 * self.rng.random()
        time.sleep(delay)
        idx = self.rng.randint(len(self.responses))
        return self.responses[idx]


# ---------------------------------------------------------------------------
# Session directories
# ---------------------------------------------------------------------------
SESSIONS_DIR = Path.home() / ".icm_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
AUTOSAVE_PATH = SESSIONS_DIR / "autosave.pkl"

SESSION_ID = "default"


# ---------------------------------------------------------------------------
# Chat application
# ---------------------------------------------------------------------------
class CliChat:
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        load_path: Optional[str] = None,
        memory_backend: str = "flat",
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.memory_backend = memory_backend
        self.scale: int = 0
        self.timing_history: list[dict[str, Any]] = []
        self.real_llm: bool = False
        self._llm_backend = None

        self._init_backend(load_path)
        self._register_atexit()

    def _get_memory(self):
        if self.real_llm:
            session = self._llm_backend._sessions.get(SESSION_ID)
            if session is None:
                self._llm_backend.create_session(SESSION_ID)
                session = self._llm_backend._sessions[SESSION_ID]
            return session["memory"]
        return self._llm_backend.memory

    def _get_history(self):
        if self.real_llm:
            session = self._llm_backend._sessions.get(SESSION_ID)
            if session is None:
                return []
            return session["history"]
        return self._llm_backend._history

    def _init_backend(self, load_path: Optional[str] = None):
        # Only try IcmLlm if --model was explicitly specified
        if self.model_name is not None:
            try:
                from hyper_ssm.llm_integration import IcmLlm
                print(c(f"  Initializing IcmLlm (model={self.model_name}) ...", _YELLOW), end="")
                sys.stdout.flush()
                t0 = time.perf_counter()
                self._llm_backend = IcmLlm(model_name=self.model_name, memory_backend=self.memory_backend)
                self._llm_backend.create_session(SESSION_ID)
                dt = time.perf_counter() - t0
                print(c(f" done in {dt:.1f}s", _GREEN))
                self.real_llm = True
            except ImportError:
                print(c("  hyper_ssm.llm_integration not available, using simulated mode", _YELLOW))
            except Exception as e:
                print(c(f"  IcmLlm init failed: {e}", _RED))
                print(c("  Falling back to simulated mode", _YELLOW))
                self.real_llm = False

        if not self.real_llm:
            print(c("  Creating InfiniteContextLLM (simulated mode, no real LLM)", _YELLOW))
            self._llm_backend = InfiniteContextLLM(embedding_dim=384, state_dim=64, num_scales=4)
            self._llm_backend._llm_fn = SimulatedLLM(embedding_dim=384)

        if load_path:
            self._load_session(load_path)
        else:
            autosave = AUTOSAVE_PATH
            if autosave.exists():
                try:
                    if self.real_llm:
                        self._llm_backend.load_session(SESSION_ID, str(autosave))
                    else:
                        self._llm_backend.load_session(str(autosave))
                    print(c(f"  Auto-loaded last session ({len(self._get_history())} messages)", _GREEN))
                except Exception:
                    pass

    def _register_atexit(self):
        atexit.register(self._autosave)

    def _autosave(self):
        try:
            if self.real_llm:
                self._llm_backend.save_session(SESSION_ID, str(AUTOSAVE_PATH))
            else:
                self._llm_backend.save_session(str(AUTOSAVE_PATH))
        except Exception:
            pass

    def _save_session(self, path: str):
        if self.real_llm:
            self._llm_backend.save_session(SESSION_ID, path)
        else:
            self._llm_backend.save_session(path)
        print(c(f"  Session saved to {path}", _GREEN))

    def _load_session(self, path: str):
        if self.real_llm:
            self._llm_backend.load_session(SESSION_ID, path)
        else:
            self._llm_backend.load_session(path)
        print(c(f"  Session loaded from {path} ({len(self._get_history())} messages)", _GREEN))

    # -----------------------------------------------------------------------
    # Commands
    # -----------------------------------------------------------------------
    @staticmethod
    def cmd_help():
        print(c(f"""
  {_BOLD}Commands:{_RESET}
    {_CYAN}/help{_RESET}              Show this help
    {_CYAN}/info{_RESET}              Show memory state info
    {_CYAN}/save <filename>{_RESET}   Save session to file
    {_CYAN}/load <filename>{_RESET}   Load session from file
    {_CYAN}/reset{_RESET}             Reset conversation (clear memory)
    {_CYAN}/history{_RESET}           Show recent messages
    {_CYAN}/model <name>{_RESET}      Switch LLM model (requires restart)
    {_CYAN}/scale <0-3>{_RESET}       Set recall scale for context building
    {_CYAN}/stats{_RESET}             Show detailed performance stats
    {_CYAN}/quit{_RESET} or {_CYAN}/exit{_RESET}   Exit
""", _YELLOW))

    def cmd_info(self):
        mem = self._get_memory()
        info = mem.info()
        print(c(f"""
  {_BOLD}Memory State{_RESET}
    Messages compressed:  {info['utterance_count']}
    State dimension:      {info['state_dim']}
    Memory size:          {info['memory_bytes']} B
    Number of scales:     {info['num_scales']}
    On manifold:          {info['state_on_manifold']}
    Current scale:        {self.scale}
    LLM type:             {'Real (HuggingFace)' if self.real_llm else 'Simulated'}
""", _MAGENTA))

    def cmd_save(self, filename: str):
        path = Path(filename).expanduser().resolve()
        self._save_session(str(path))

    def cmd_load(self, filename: str):
        path = Path(filename).expanduser().resolve()
        if not path.exists():
            print(c(f"  File not found: {path}", _RED))
            return
        self._load_session(str(path))

    def cmd_reset(self):
        if self.real_llm:
            self._llm_backend.delete_session(SESSION_ID)
            self._llm_backend.create_session(SESSION_ID)
        else:
            self._llm_backend.memory.reset()
            self._llm_backend._history.clear()
        self.timing_history.clear()
        print(c("  Conversation reset. Memory cleared.", _GREEN))

    def cmd_history(self):
        history = self._get_history()
        if not history:
            print(c("  No messages yet.", _YELLOW))
            return
        n = len(history)
        start = max(0, n - 20)
        for i in range(start, n):
            msg = history[i]
            role = msg["role"]
            content = msg["content"][:200]
            if role == "user":
                print(c(f"  [{i}] You: {content}", _CYAN))
            else:
                print(c(f"  [{i}] Assistant: {content}", _GREEN))

    @staticmethod
    def cmd_model(name: str):
        print(c(f"  Model switching requires restart. "
                f"Please exit and re-run with --model \"{name}\"", _YELLOW))

    def cmd_scale(self, arg: str):
        try:
            val = int(arg)
            if 0 <= val <= 3:
                self.scale = val
                print(c(f"  Recall scale set to {val}", _GREEN))
            else:
                print(c("  Scale must be 0-3", _RED))
        except ValueError:
            print(c("  Usage: /scale <0-3>", _RED))

    def cmd_stats(self):
        if not self.timing_history:
            print(c("  No stats available yet.", _YELLOW))
            return
        times = [t["gen_time"] for t in self.timing_history]
        tokens = [t.get("tokens", 0) for t in self.timing_history]
        mem = self._get_memory()
        print(c(f"""
  {_BOLD}Performance Stats{_RESET}
    Total turns:         {mem._utterance_count}
    Memory bytes:        {mem.memory_size_bytes} B
    Avg generation time: {np.mean(times):.3f}s
    Total gen time:      {np.sum(times):.3f}s
    Min/Max gen time:    {np.min(times):.3f}s / {np.max(times):.3f}s
    Messages in history: {len(self._get_history())}
    On manifold:         {mem.info()['state_on_manifold']}
""", _MAGENTA))

    def cmd_quit(self):
        print(c("\n  Saving session before exit...", _YELLOW))
        self._autosave()
        print(c("  Goodbye!", _GREEN))
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Message loop
    # -----------------------------------------------------------------------
    def chat_turn(self, user_message: str) -> tuple[str, float, float]:
        t0 = time.perf_counter()

        if self.real_llm:
            response = self._llm_backend.chat(SESSION_ID, user_message)
        else:
            response = self._llm_backend.chat(user_message)

        gen_time = time.perf_counter() - t0
        n_tokens = len(response.split())
        tokens_per_sec = n_tokens / gen_time if gen_time > 0 else 0

        self.timing_history.append({
            "gen_time": gen_time,
            "tokens": n_tokens,
            "tps": tokens_per_sec,
        })

        return response, gen_time, tokens_per_sec

    def run(self):
        print(BANNER)
        mem = self._get_memory()
        print(c(f"  Model:      {_BOLD}{self.model_name or 'simulated'}{_RESET}", _WHITE))
        print(c(f"  Memory:     {_BOLD}InfiniteContextMemory{_RESET} "
                f"(dim={mem.state_dim}, scales={mem.num_scales})", _WHITE))
        print(c(f"  O(1) claim: {_BOLD}State size fixed at "
                f"{(mem.state_dim + 1) * 4}B{_RESET} "
                f"regardless of conversation length", _WHITE))
        print(c(f"  Sessions:   {_BOLD}{SESSIONS_DIR}{_RESET}", _WHITE))
        print(c(f"  Type {_CYAN}/help{_RESET} for commands, "
                f"{_CYAN}/quit{_RESET} to exit.", _DIM))
        print()

        turn_number = len(self._get_history()) // 2

        completer = Completer()
        if readline is not None:
            readline.set_completer(completer.complete)
            readline.parse_and_bind("tab: complete")
            readline.set_history_length(500)

        while True:
            try:
                mem = self._get_memory()
                prompt = (
                    c(f"\n  [{_BOLD}{turn_number + 1}{_RESET}] "
                      f"{_DIM}Memory: {mem.memory_size_bytes}B | "
                      f"Turns: {mem._utterance_count}{_RESET}", _WHITE)
                    + c("\n  You: ", _CYAN)
                )
                user_message = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self.cmd_quit()
                break

            if not user_message:
                continue

            if user_message.startswith("/"):
                parts = user_message.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    self.cmd_quit()
                elif cmd == "/help":
                    self.cmd_help()
                elif cmd == "/info":
                    self.cmd_info()
                elif cmd == "/save":
                    self.cmd_save(arg)
                elif cmd == "/load":
                    self.cmd_load(arg)
                elif cmd == "/reset":
                    self.cmd_reset()
                elif cmd == "/history":
                    self.cmd_history()
                elif cmd == "/model":
                    self.cmd_model(arg)
                elif cmd == "/scale":
                    self.cmd_scale(arg)
                elif cmd == "/stats":
                    self.cmd_stats()
                else:
                    print(c(f"  Unknown command: {cmd}. "
                            f"Type /help for available commands.", _RED))
                continue

            print(c("  Assistant: ", _GREEN), end="", flush=True)
            response, gen_time, tps = self.chat_turn(user_message)
            wrapper = textwrap.TextWrapper(
                width=70, initial_indent="", subsequent_indent="             "
            )
            for line in response.split("\n"):
                for wrapped_line in wrapper.wrap(line):
                    print(c(wrapped_line, _GREEN))
            print(c(f"  {_DIM}({gen_time:.2f}s  {tps:.0f} tok/s){_RESET}", _MAGENTA))

            mem = self._get_memory()
            if mem._utterance_count > 0 and mem._utterance_count % 10 == 0:
                self._autosave()

            turn_number += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Infinite Context Memory -- Interactive CLI Chat"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name "
                             "(e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Sentence-transformer model for embeddings")
    parser.add_argument("--load", type=str, default=None,
                        help="Load session from pickle file")
    parser.add_argument("--memory-backend", type=str, default="flat", choices=["flat", "tree"],
                        help="Memory backend: flat (O(1)) or tree (O(log N))")
    args = parser.parse_args()

    app = CliChat(
        model_name=args.model,
        embedding_model_name=args.embedding_model,
        load_path=args.load,
        memory_backend=args.memory_backend,
    )
    try:
        app.run()
    except KeyboardInterrupt:
        print()
        app.cmd_quit()


if __name__ == "__main__":
    main()
