#!/usr/bin/env python3
# ruff: noqa: ASYNC230  (file writes happen after measurement, nothing runs concurrently)
"""Compare chat models on latency and stability, against the real corpus.

Fills in the "Expected speed" column of the hardware table in CLAUDE.md, which
has said "not yet benchmarked on this hardware" since the model was switched.

What it measures
────────────────
Latency, per model:
  * TTFT      — time to first visible token. This is what the user actually
                perceives as "did it hang?", and it is where a model that
                secretly reasons before answering gets caught.
  * tok/s     — streaming throughput once generation starts.
  * total     — wall-clock to a finished answer.
  * load      — resident size reported by Ollama, so you can see whether two
                models can stay loaded together in 16 GB.

Stability, per model — no labelled answers required:
  * numeric drift — every number+unit in each answer is extracted and compared
                across repeat runs of the identical question. A model that says
                "28 V" once and "24 V" the next time is unreliable regardless of
                which is correct. This is the failure mode that disqualified
                llama3.1:8b (self-contradicted with multiple wrong values in one
                answer) and llama3.2 (hallucinated values).
  * refusal rate — how often the model falls back to "not in the ingested
                documents" on a question its peers answer. High refusal is not
                automatically bad (it is the honest response for a bad
                retrieval), but a model refusing where others answer is a signal.
  * length cv — coefficient of variation of answer length across runs.

Retrieval runs ONCE per question and the identical context is replayed to every
model, so the numbers reflect the model and not retrieval jitter.

Usage
─────
    # Ollama and Qdrant must be running, with documents ingested.
    python3 benchmark_models.py --models qwen3.5:4b llama3.1:8b
    python3 benchmark_models.py --models qwen3.5:4b --runs 5
    python3 benchmark_models.py --questions my_questions.txt --out results.md

Models are pulled if missing. On CPU expect a few minutes per model per run —
start with --runs 3 and two models before running a wide sweep.
"""
import argparse
import asyncio
import json
import re
import statistics
import time

import config
import llm
import retriever
import store

# Generic questions that exercise different retrieval paths. Override with
# --questions to use ones specific to your corpus.
DEFAULT_QUESTIONS = [
    "What is the maximum operating temperature?",
    "What voltage does the system require?",
    "List the main components described in the documentation.",
    "What are the installation prerequisites?",
    "Summarise the safety warnings.",
]

# number + optional unit, e.g. "28 V", "59.473 kg", "-40°C", "1.2M"
_VALUE_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*"
    r"(°?[CFK]\b|V\b|A\b|W\b|Hz\b|kHz\b|MHz\b|GHz\b|kg\b|g\b|mm\b|cm\b|m\b|"
    r"bar\b|psi\b|Nm\b|%|s\b|ms\b|min\b|h\b)?"
)

_REFUSAL = "not in the ingested documents"


def extract_values(text: str) -> set[str]:
    """Normalised {number+unit} set, for comparing answers across runs."""
    out = set()
    for num, unit in _VALUE_RE.findall(text):
        try:
            n = float(num)
        except ValueError:
            continue
        out.add(f"{n:g}{(unit or '').strip().lower()}")
    return out


async def ensure_model(name: str) -> bool:
    """Pull the model if Ollama does not have it. Returns False if unavailable."""
    try:
        r = await llm._client.get("/api/tags")
        have = {m["name"] for m in r.json().get("models", [])}
    except Exception as exc:
        print(f"  ! cannot reach Ollama: {exc}")
        return False
    if any(h == name or h.startswith(name.split(":")[0] + ":") for h in have):
        return True
    print(f"  pulling {name} (first run only) ...")
    try:
        r = await llm._client.post("/api/pull", json={"model": name, "stream": False}, timeout=None)
        return r.is_success
    except Exception as exc:
        print(f"  ! pull failed: {exc}")
        return False


async def model_footprint(name: str) -> str:
    """Resident size from Ollama, so you can see if two models fit in RAM together."""
    try:
        r = await llm._client.get("/api/ps")
        for m in r.json().get("models", []):
            if m.get("name", "").startswith(name.split(":")[0]):
                return f"{m.get('size', 0) / 1e9:.1f} GB"
    except Exception:
        pass
    return "—"


async def timed_generation(system: str, question: str, model: str) -> dict:
    """One streamed generation, instrumented."""
    start = time.perf_counter()
    ttft = None
    tokens = 0
    parts: list[str] = []
    async for token in llm.chat_stream(system=system, user=question, model=model):
        if ttft is None:
            ttft = time.perf_counter() - start
        tokens += 1
        parts.append(token)
    total = time.perf_counter() - start
    answer = "".join(parts)
    gen_time = max(total - (ttft or 0), 1e-6)
    return {
        "ttft": ttft or total,
        "total": total,
        "tok_s": tokens / gen_time,
        "answer": answer,
        "values": extract_values(answer),
        "refused": _REFUSAL in answer.lower(),
        "chars": len(answer),
    }


def _cv(xs: list[float]) -> float:
    """Coefficient of variation — stddev relative to mean."""
    xs = [x for x in xs if x]
    if len(xs) < 2:
        return 0.0
    m = statistics.mean(xs)
    return (statistics.pstdev(xs) / m) if m else 0.0


def drift_score(runs: list[dict]) -> float:
    """Fraction of extracted values that are NOT present in every run.

    0.0 means every run produced an identical set of numbers. 1.0 means no
    number was reproduced. Only meaningful with --runs >= 2.
    """
    sets = [r["values"] for r in runs if not r["refused"]]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    union = set().union(*sets)
    stable = set.intersection(*sets)
    return 0.0 if not union else 1.0 - len(stable) / len(union)


async def benchmark(models: list[str], questions: list[str], runs: int) -> dict:
    print(f"Retrieving context for {len(questions)} question(s) ...")
    contexts: dict[str, str] = {}
    for q in questions:
        result = await retriever._retrieve(q)
        if result is None:
            print(f"  ! no context retrieved for {q!r} — skipping")
            continue
        context_parts, _sources = result
        # Identical prompt for every model, so we measure the model alone.
        contexts[q] = retriever._build_system(context_parts, None)
    if not contexts:
        raise SystemExit("No context retrieved for any question — is anything ingested?")

    results: dict[str, dict] = {}
    for model in models:
        print(f"\n=== {model} ===")
        if not await ensure_model(model):
            print("  skipped (unavailable)")
            continue

        per_question: dict[str, list[dict]] = {}
        for q, system in contexts.items():
            print(f"  {q[:60]!r}")
            attempts = []
            for i in range(runs):
                try:
                    r = await timed_generation(system, q, model)
                except Exception as exc:
                    print(f"    run {i + 1}: FAILED ({exc})")
                    continue
                attempts.append(r)
                print(
                    f"    run {i + 1}: ttft={r['ttft']:.1f}s total={r['total']:.1f}s "
                    f"{r['tok_s']:.1f} tok/s{' [refused]' if r['refused'] else ''}"
                )
            if attempts:
                per_question[q] = attempts

        footprint = await model_footprint(model)
        # Evict before moving to the next model. Without this a sweep across
        # several models leaves every one of them resident, which on a 16 GB
        # machine means swapping long before the sweep finishes.
        await llm.unload(model)
        print(f"  unloaded {model}")

        if not per_question:
            continue
        flat = [r for rs in per_question.values() for r in rs]
        results[model] = {
            "ttft_med": statistics.median(r["ttft"] for r in flat),
            "total_med": statistics.median(r["total"] for r in flat),
            "tok_s_med": statistics.median(r["tok_s"] for r in flat),
            "refusal_rate": sum(r["refused"] for r in flat) / len(flat),
            "drift": statistics.mean(drift_score(rs) for rs in per_question.values()),
            "len_cv": statistics.mean(_cv([r["chars"] for r in rs]) for rs in per_question.values()),
            "footprint": footprint,
            "n": len(flat),
            "raw": {q: [{k: v for k, v in r.items() if k != "values"} for r in rs]
                    for q, rs in per_question.items()},
        }
    return results


def render(results: dict, runs: int) -> str:
    lines = [
        "# Chat model benchmark",
        "",
        f"Latency and stability only — no accuracy scoring. {runs} run(s) per question.",
        "Identical retrieved context replayed to every model.",
        "",
        "| Model | TTFT (med) | Total (med) | tok/s | Resident | Numeric drift | Refusal | Length CV |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, m in sorted(results.items(), key=lambda kv: kv[1]["total_med"]):
        lines.append(
            f"| `{model}` | {m['ttft_med']:.1f}s | {m['total_med']:.1f}s | "
            f"{m['tok_s_med']:.1f} | {m['footprint']} | {m['drift']:.2f} | "
            f"{m['refusal_rate']:.0%} | {m['len_cv']:.2f} |"
        )
    lines += [
        "",
        "**Numeric drift** — fraction of numbers that did not reproduce across repeat",
        "runs of the same question. 0.00 is perfectly reproducible; anything above",
        "~0.2 means the model gives different values for the same question and should",
        "not be trusted for spec lookups regardless of its speed.",
        "",
        "**Refusal** — share of answers that fell back to \"not in the ingested",
        "documents\". Compare across models on identical context: a model refusing",
        "where others answer is under-using the context it was given.",
        "",
        "**Length CV** — variation in answer length across runs. High values alongside",
        "low drift usually just means verbosity varies, which is harmless.",
        "",
        "A model is only a candidate if drift is near zero. Speed is the tiebreak,",
        "not the criterion — the previous default was replaced for wrong values, not",
        "for being slow.",
    ]
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark chat models on latency and stability")
    ap.add_argument("--models", nargs="+", default=[config.CHAT_MODEL],
                    help="Ollama model tags to compare")
    ap.add_argument("--runs", type=int, default=3,
                    help="Repeats per question; >=2 required for drift (default 3)")
    ap.add_argument("--questions", help="File with one question per line")
    ap.add_argument("--out", default="benchmark_results.md", help="Markdown output path")
    ap.add_argument("--json", help="Also write raw timings as JSON")
    args = ap.parse_args()

    if args.questions:
        with open(args.questions) as f:
            questions = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    else:
        questions = DEFAULT_QUESTIONS

    if await store.count() == 0:
        raise SystemExit("Nothing ingested — ingest documents before benchmarking.")

    results = await benchmark(args.models, questions, args.runs)
    if not results:
        raise SystemExit("No model produced results.")

    report = render(results, args.runs)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("\n" + report)
    print(f"\nWritten to {args.out}")
    await store.aclose()
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
