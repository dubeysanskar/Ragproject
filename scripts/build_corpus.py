"""Build the GoaRAG corpus from ai4bharat/MSMARCO-XI (PDR §3).

The dataset is 55 GB across 27 parquet files, so this never downloads it.
Instead it streams row batches over HTTP from the validation split's parquet
(one file per target language, 97,941 rows each) and stops after --rows-per-lang.

Output:
  data/corpus.json   — passages the API indexes at boot
  data/eval.json     — held-out queries with gold answers/passages for bench.py

The English corpus needs no separate source: every row carries the original
`English_passages` and `Eng_Query` alongside the translation, so English is
extracted from the same rows as Hindi (deduped by query_id across languages).

Usage (from raghhgtask2/):
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe scripts/build_corpus.py \
      --langs hin_Deva tam_Taml --rows-per-lang 400 --eval-per-lang 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

BASE = "datasets/ai4bharat/MSMARCO-XI@refs%2Fconvert%2Fparquet/default/validation"
COLS = ["target_lang", "query", "Eng_Query", "Answer", "Eng_Answer", "passages", "query_id"]

# ISO tag the pipeline's langdetect produces, per dataset language code.
LANG_TAG = {
    "asm_Beng": "bn", "ben_Beng": "bn", "guj_Gujr": "gu", "hin_Deva": "hi",
    "kan_Knda": "kn", "mal_Mlym": "ml", "mar_Deva": "hi", "npi_Deva": "hi",
    "ory_Orya": "or", "pan_Guru": "pa", "san_Deva": "hi", "tam_Taml": "ta",
    "tel_Telu": "te", "urd_Arab": "ur",
}


def locate_language_files(fs: HfFileSystem, wanted: set[str]) -> dict[str, str]:
    """Each validation file holds exactly one language; probe the first row of
    each file's `target_lang` column until every wanted language is found."""
    found: dict[str, str] = {}
    for name in sorted(fs.ls(BASE, detail=False)):
        if len(found) == len(wanted):
            break
        with fs.open(name, "rb") as f:
            p = pq.ParquetFile(f)
            batch = next(p.iter_batches(batch_size=1, columns=["target_lang"]))
            lang = batch.column(0)[0].as_py()
        print(f"  {name.split('/')[-1]} -> {lang}")
        if lang in wanted:
            found[lang] = name
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"languages not present in validation split: {missing}")
    return found


def harvest(
    fs: HfFileSystem,
    path: str,
    dataset_lang: str,
    n_rows: int,
    n_eval: int,
    want_english: bool,
    seen_query_ids: set[int],
) -> tuple[list[dict], list[dict]]:
    """Stream one language file. Returns (passages, eval_queries).

    Corpus rule: index the *selected* passage (the one MS MARCO marked as
    actually answering) plus the first distractor, so retrieval has to beat
    plausible-but-wrong neighbours — a corpus of only gold passages would make
    Recall@5 meaninglessly easy.
    """
    tag = LANG_TAG[dataset_lang]
    passages: list[dict] = []
    evals: list[dict] = []
    t0 = time.perf_counter()

    with fs.open(path, "rb") as f:
        p = pq.ParquetFile(f)
        for batch in p.iter_batches(batch_size=256, columns=COLS):
            for r in batch.to_pylist():
                if len(passages) >= n_rows * 2 and len(evals) >= n_eval:
                    break
                qid = r["query_id"]
                if qid in seen_query_ids:
                    continue
                seen_query_ids.add(qid)

                sel = r["passages"]["is_selected"]
                translated = r["passages"]["Translated_passages"]
                english = r["passages"]["English_passages"]
                try:
                    gold_i = sel.index(1)
                except ValueError:
                    continue  # no annotated answer passage — useless for eval
                distractor_i = next((i for i in range(len(sel)) if not sel[i]), None)

                # Held-out rows must not have their query indexed as a C5 key,
                # or the eval query matches a verbatim copy of itself (cosine
                # 1.0) and every retrieval metric measures leakage, not skill.
                held_out = len(evals) < n_eval

                def add(texts: list[str], lang: str, query: str, answer: str | None, suffix: str):
                    gold_pid = f"{qid}-{suffix}-g"
                    passages.append({
                        "passage_id": gold_pid,
                        "text": texts[gold_i],
                        "lang": lang,
                        "questions": [] if held_out else [query],
                    })
                    if distractor_i is not None:
                        passages.append({
                            "passage_id": f"{qid}-{suffix}-d",
                            "text": texts[distractor_i],
                            "lang": lang,
                            "questions": [],
                        })
                    if len(evals) < n_eval and answer:
                        evals.append({
                            "query_id": qid,
                            "query": query,
                            "lang": lang,
                            "gold_passage_id": gold_pid,
                            "gold_answer": answer,
                        })

                if len(passages) < n_rows:
                    add(translated, tag, r["query"], r["Answer"], tag)
                if want_english and len(passages) < n_rows * 2:
                    add(english, "en", r["Eng_Query"], r["Eng_Answer"], "en")
            else:
                continue
            break

    print(f"  {dataset_lang}: {len(passages)} passages, {len(evals)} eval "
          f"({time.perf_counter() - t0:.0f}s)")
    return passages, evals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hin_Deva", "tam_Taml"],
                    help="dataset language codes; English rides along free")
    ap.add_argument("--rows-per-lang", type=int, default=400)
    ap.add_argument("--eval-per-lang", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    fs = HfFileSystem()
    print("locating language files...")
    files = locate_language_files(fs, set(args.langs))

    all_passages: list[dict] = []
    all_evals: list[dict] = []
    seen: set[int] = set()
    for i, (lang, path) in enumerate(files.items()):
        print(f"harvesting {lang} ...")
        # English passages come from the first language's rows only, so the
        # same query_id never lands twice via two translations.
        ps, ev = harvest(fs, path, lang, args.rows_per_lang, args.eval_per_lang,
                         want_english=(i == 0), seen_query_ids=seen)
        all_passages.extend(ps)
        all_evals.extend(ev)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "corpus.json").write_text(
        json.dumps(all_passages, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "eval.json").write_text(
        json.dumps(all_evals, ensure_ascii=False, indent=1), encoding="utf-8")
    langs = {}
    for p in all_passages:
        langs[p["lang"]] = langs.get(p["lang"], 0) + 1
    print(f"\nwrote {len(all_passages)} passages {langs} + {len(all_evals)} eval queries -> {args.out}/")


if __name__ == "__main__":
    main()
