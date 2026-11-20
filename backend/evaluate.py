"""Run: python evaluate.py   -> retrieval hit-rate, faithfulness, correct refusals."""
import json
from pathlib import Path

import loaders
from chatbot import FALLBACK, Chatbot
from kb import KnowledgeBase

kb = KnowledgeBase(namespace="eval", db_path="data/eval.db", storage_dir="storage_eval")
kb.reset()
for p in sorted(Path("data/sample_docs").glob("*")):
    if p.suffix.lower() in loaders.SUPPORTED_EXTS:
        print("indexing", p.name, kb.add_file(str(p), p.name))
bot = Chatbot(namespace="eval", db_path="data/eval.db")

cases = json.loads(Path("data/eval_set.json").read_text())
hits = faithful = answerable = refused_ok = unanswerable = 0
for c in cases:
    res = bot.ask(c["question"])
    if c.get("expected_source"):
        answerable += 1
        hit = c["expected_source"] in {s["source"] for s in res["sources"]}
        hits += hit
        ok = hit and (not res["docs"] or bot.judge_faithfulness(res["answer"], res["docs"]))
        faithful += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {c['question']}")
    else:
        unanswerable += 1
        ok = FALLBACK in res["answer"]
        refused_ok += ok
        print(f"[{'PASS' if ok else 'FAIL'}] (out-of-scope) {c['question']}")

print("\n--- RESULTS ---")
print(f"Retrieval hit rate : {hits}/{answerable}")
print(f"Faithful answers   : {faithful}/{answerable}")
print(f"Correct refusals   : {refused_ok}/{unanswerable}")
