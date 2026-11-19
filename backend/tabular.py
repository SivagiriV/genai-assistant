"""Exact answers over uploaded CSV / Excel tables (counts, breakdowns, row lookups).

The LLM only produces a small JSON *plan*; we validate it and run it with pandas.
No LLM-generated code is ever executed.
"""
import json
import re
from pathlib import Path

import pandas as pd


class TableQA:
    def __init__(self, llm):
        self.llm = llm
        self.tables = {}

    # ---------- table management ----------
    def add(self, path, name=None):
        p = Path(path)
        name = name or p.name
        ext = Path(name).suffix.lower() or p.suffix.lower()
        if ext == ".csv":
            try:
                self.tables[name] = pd.read_csv(p)
            except UnicodeDecodeError:
                self.tables[name] = pd.read_csv(p, encoding="latin-1")
        elif ext == ".xlsx":
            for sheet, df in pd.read_excel(p, sheet_name=None).items():
                self.tables[f"{name} [{sheet}]"] = df

    def remove(self, name):
        for k in [k for k in self.tables if k == name or k.startswith(f"{name} [")]:
            del self.tables[k]

    def clear(self):
        self.tables.clear()

    def _schema(self):
        lines = []
        for name, df in self.tables.items():
            cols = []
            for c in df.columns:
                if df[c].nunique() <= 12:
                    vals = ", ".join(map(str, df[c].dropna().unique()))
                    cols.append(f"{c} (values: {vals})")
                else:
                    ex = ", ".join(str(v)[:30] for v in df[c].dropna().head(2))
                    cols.append(f"{c} (e.g. {ex})")
            lines.append(f"- {name} ({len(df)} rows): " + "; ".join(cols))
        return "\n".join(lines)

    # ---------- planning ----------
    def _plan(self, question):
        prompt = (
            "You route questions about uploaded tables.\nTables:\n" + self._schema() + "\n\n"
            "If the question asks for a COUNT, TOTAL, BREAKDOWN or DISTRIBUTION over rows of a table, "
            "or asks about a specific row/ticket/ID, reply with JSON only:\n"
            '{"type":"aggregate","table":"<table name exactly as listed>","operation":"count" or "count_by" or "show",'
            '"group_by":"<column or null>","filters":[{"column":"<col>","op":"eq" or "contains" or "endswith","value":"<text>"}]}\n'
            '"show" returns the matching rows themselves (use it when the user asks about a specific '
            'ticket/row/ID, e.g. "details of ticket 237"). If the user gives only a number for an ID column '
            'whose values look like TKT-10237, use op "endswith" with that number.\n'
            'For any other question (explanations, policies, how-to) reply {"type":"lookup"}.\n'
            "Reply with JSON only, no markdown.\n\nQuestion: " + question)
        raw = self.llm.invoke(prompt).content
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            return json.loads(raw)
        except Exception:
            return {"type": "lookup"}

    # ---------- execution ----------
    def try_answer(self, question):
        """Return {'answer','sources'} for table questions, else None (use normal RAG)."""
        if not self.tables:
            return None
        plan = self._plan(question)
        if plan.get("type") != "aggregate":
            return None
        name = plan.get("table")
        if name not in self.tables:
            if len(self.tables) == 1:
                name = next(iter(self.tables))
            else:
                return None
        df = self.tables[name]
        total = len(df)
        desc = []
        for f in plan.get("filters") or []:
            col, op, val = f.get("column"), f.get("op"), str(f.get("value", ""))
            if col not in df.columns:
                return None
            s = df[col].astype(str)
            if op == "eq":
                df = df[s.str.lower() == val.lower()]
            elif op == "endswith":
                df = df[s.str.lower().str.endswith(val.lower())]
            elif op == "contains":
                df = df[s.str.contains(val, case=False, regex=False)]
            else:
                return None
            desc.append(f"{col} {op} '{val}'")
        where = f" where {' and '.join(desc)}" if desc else ""
        src = [{"source": name, "page": None, "url": None,
                "text": f"Computed with pandas over all {total} rows (not a sample)."}]

        op = plan.get("operation")
        nice = lambda c: str(c).replace("_", " ").title()        # ticket_id -> Ticket Id
        if op == "show":
            if df.empty:
                return {"answer": "I couldn't find any matching records.", "sources": []}
            blocks = ["\n".join(f"- **{nice(c)}**: {v}" for c, v in row.items())
                      for _, row in df.head(10).iterrows()]
            head = "Here's what I found:" if len(df) == 1 else f"I found **{len(df)}** matching records:"
            note = f"\n\n_Showing the first 10 of {len(df)}._" if len(df) > 10 else ""
            return {"answer": head + "\n\n" + "\n\n".join(blocks) + note, "sources": src}
        if op == "count":
            if desc:
                answer = f"There are **{len(df)}** matching records (out of {total} in total)."
            else:
                answer = f"There are **{len(df)}** records in total."
            return {"answer": answer, "sources": src}
        if op == "count_by" and plan.get("group_by") in df.columns:
            g = plan["group_by"]
            body = "\n".join(f"- {k}: **{v}**" for k, v in df[g].value_counts().items())
            scope = f" (for the {len(df)} matching records)" if desc else ""
            return {"answer": f"Here is the breakdown by **{nice(g)}**{scope}:\n\n{body}", "sources": src}
        return None