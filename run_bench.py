"""One-shot launcher: heuristic benchmark with lawnsonic-bench identity."""
import os, sys
os.environ.setdefault("AGENT_MODE", "heuristic")
os.environ.setdefault("ARENA_COMPETITION_ID", "seed_poker_eval_s1")
sys.path.insert(0, "examples")
from agent import main
sys.exit(main([
    "--handle", "lawnsonic-bench",
    "--name",   "Lawnsonic Bench",
    "--quote",  "math is the only bluff that scales",
]))
