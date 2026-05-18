#!/usr/bin/env python3
# Convert ds4_eval.c's eval_cases[] (C99 array-element designators) into a form
# that compiles as C++ (collapse all .choice[N] = "X" lines per case into one
# .choice = { "X0", "X1", ..., nullptr } line).
import re, sys, pathlib

p = pathlib.Path(__file__).resolve().parent / "eval_cases_data.inc"
src = p.read_text()

# Split on top-level "    },\n" between cases.  Simpler: process inside each
# brace-delimited case independently.
out = []
i = 0
n = len(src)
in_case = False
case_buf = []

choice_line = re.compile(r'\s*\.choice\[(\d+)\] = (".*"),?\s*$')

def flush_case(lines):
    choices = {}
    other = []
    first_choice_index = None
    for idx, ln in enumerate(lines):
        m = choice_line.match(ln)
        if m:
            choices[int(m.group(1))] = m.group(2)
            if first_choice_index is None:
                first_choice_index = idx
        else:
            other.append((idx, ln))
    # rebuild
    new = []
    inserted = False
    for idx, ln in other:
        if not inserted and first_choice_index is not None and idx > first_choice_index:
            items = []
            for k in sorted(choices):
                items.append(choices[k])
            # only add nullptr terminator if there's room (EVAL_MAX_CHOICES = 10)
            if len(items) < 10:
                items.append("nullptr")
            indent = " " * 8
            new.append(f"{indent}.choice = {{ {', '.join(items)} }},")
            inserted = True
        new.append(ln)
    if not inserted and choices:
        items = [choices[k] for k in sorted(choices)]
        if len(items) < 10:
            items.append("nullptr")
        new.append(f"        .choice = {{ {', '.join(items)} }},")
    return new

# Walk line by line tracking { ... } depth >= 2 = inside a case.
lines = src.splitlines()
result = []
case_lines = None
depth = 0
for ln in lines:
    stripped = ln.strip()
    if depth == 1 and stripped == "{":
        # entering a case
        case_lines = []
        result.append(ln)
        depth = 2
        continue
    if depth == 2:
        if stripped.startswith("},"):
            # flush
            result.extend(flush_case(case_lines))
            result.append(ln)
            case_lines = None
            depth = 1
            continue
        case_lines.append(ln)
        continue
    # depth 0 or 1
    result.append(ln)
    # update depth
    for ch in ln:
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1

p.write_text("\n".join(result) + "\n")
print(f"rewrote {p}")
