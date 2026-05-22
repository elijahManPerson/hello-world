"""
run_all.py
==========
One script that runs the whole pipeline in the correct order.

It expects these files sitting next to it, with these exact names:
    aes_canonical.py        (your original, unchanged)
    corrector_claude.py     (your original corrector)
    mech_v2.py              (Changes 1, 2, 4, 7)
    engine_prototype.py     (Changes 3, 5, 6)
    concern1_detector.py
    title_detector.py
    concurrency.py

Changes applied to this file: 8 (empty-correction guard + retry).
Token and time tracking built in; summaries saved to out_dir on every run.

Edit the CONFIG block, then run:  python3 run_all.py

Each engine stage has its own switch. Set a stage to "mock" first to see
what it would do at zero cost, then "run" it for real once you are happy.
Stages already proven on data (correction is yours; the cheap layers) just
run. The order matters and is enforced here so you do not have to think
about it.
"""

import os
import re
import time as _time
import pandas as pd

# ----------------------------------------------------------------------
# Token tracking — patches anthropic.Anthropic globally so every stage's
# API calls are captured regardless of which module creates the client.
# Usage is accumulated in _TOKEN_LOG; call token_summary() to report.
# ----------------------------------------------------------------------
_TOKEN_STAGE = ["unknown"]   # mutable so closures can read the current stage
_TOKEN_LOG   = []            # list of per-call dicts

# ----------------------------------------------------------------------
# Time tracking — records wall-clock duration of each named stage.
# ----------------------------------------------------------------------
_TIME_LOG = {}   # {stage: {"start": float, "end": float}}


def _install_token_tracker():
    import anthropic as _ant
    _Orig = _ant.Anthropic
    class _Tracked(_Orig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            _orig_create = self.messages.create
            def _create(*args, **kwargs):
                resp = _orig_create(*args, **kwargs)
                u = getattr(resp, "usage", None)
                if u:
                    _TOKEN_LOG.append({
                        "stage":       _TOKEN_STAGE[0],
                        "model":       kwargs.get("model", args[0] if args else "?"),
                        "input":       getattr(u, "input_tokens", 0),
                        "output":      getattr(u, "output_tokens", 0),
                        "cache_read":  getattr(u, "cache_read_input_tokens", 0) or 0,
                        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                    })
                return resp
            self.messages.create = _create
    _ant.Anthropic = _Tracked

_install_token_tracker()


def _stage_start(name):
    _TOKEN_STAGE[0] = name
    _TIME_LOG[name] = {"start": _time.time(), "end": None}

def _stage_end(name):
    if name in _TIME_LOG:
        _TIME_LOG[name]["end"] = _time.time()

def _set_stage(name):
    """Start timing and set current token-tracking stage."""
    _stage_start(name)


def token_summary(out_dir=None):
    """Print and (optionally) save a per-stage token usage table."""
    if not _TOKEN_LOG:
        print("No API calls recorded.")
        return pd.DataFrame()

    df = pd.DataFrame(_TOKEN_LOG)
    grp = (df.groupby(["stage", "model"])
             .agg(calls=("input","count"),
                  input_tokens=("input","sum"),
                  output_tokens=("output","sum"),
                  cache_read_tokens=("cache_read","sum"),
                  cache_write_tokens=("cache_write","sum"))
             .reset_index())
    # cost estimate (USD) — list prices May 2025
    _PRICE = {
        "claude-opus-4-7":           (15.00, 75.00),
        "claude-sonnet-4-6":         ( 3.00, 15.00),
        "claude-haiku-4-5-20251001": ( 0.80,  4.00),
    }
    def _cost(row):
        inp_p, out_p = _PRICE.get(row["model"], (3.00, 15.00))
        # cache reads ~10% of input price; writes ~125%
        return ((row["input_tokens"]       * inp_p
               + row["output_tokens"]      * out_p
               + row["cache_read_tokens"]  * inp_p * 0.10
               + row["cache_write_tokens"] * inp_p * 1.25) / 1_000_000)
    grp["est_cost_usd"] = grp.apply(_cost, axis=1).round(4)

    tot = grp[["calls","input_tokens","output_tokens",
               "cache_read_tokens","cache_write_tokens","est_cost_usd"]].sum()
    tot_row = pd.DataFrame([{"stage":"TOTAL","model":"",
                              "calls":int(tot["calls"]),
                              "input_tokens":int(tot["input_tokens"]),
                              "output_tokens":int(tot["output_tokens"]),
                              "cache_read_tokens":int(tot["cache_read_tokens"]),
                              "cache_write_tokens":int(tot["cache_write_tokens"]),
                              "est_cost_usd":round(tot["est_cost_usd"],4)}])
    out = pd.concat([grp, tot_row], ignore_index=True)

    print("\n" + "="*72)
    print("TOKEN SUMMARY")
    print("="*72)
    print(out.to_string(index=False))
    print("="*72 + "\n")

    if out_dir:
        path = f"{out_dir}/token_summary.csv"
        out.to_csv(path, index=False)
        print(f"Token summary saved to {path}")
    return out


def time_summary(total_start, out_dir=None):
    """Print and (optionally) save a per-stage wall-clock time table."""
    rows = []
    for stage, t in _TIME_LOG.items():
        start, end = t["start"], t["end"]
        secs = (end - start) if end else (_time.time() - start)
        rows.append({"stage": stage,
                     "minutes": round(secs / 60, 2),
                     "seconds": round(secs, 1)})
    total_secs = _time.time() - total_start
    rows.append({"stage": "TOTAL",
                 "minutes": round(total_secs / 60, 2),
                 "seconds": round(total_secs, 1)})

    out = pd.DataFrame(rows)
    print("\n" + "="*50)
    print("TIME SUMMARY")
    print("="*50)
    print(out.to_string(index=False))
    print("="*50 + "\n")

    if out_dir:
        path = f"{out_dir}/time_summary.csv"
        out.to_csv(path, index=False)
        print(f"Time summary saved to {path}")
    return out


# ----------------------------------------------------------------------
# CONFIG  — the only part you edit
# ----------------------------------------------------------------------
CONFIG = {
    "input_csv":  "your_data.csv",         # must have the id + raw columns
    "id_col":     "Research ID",
    "raw_col":    "Raw text",
    "corr_col":   "Corrected text (8)",    # created if not already present
    "out_dir":    "./aes_output",

    "api_key":    os.environ.get("ANTHROPIC_API_KEY", ""),

    # per-stage: "run", "mock", or "skip"
    "stage_correct":        "run",         # Opus, your existing corrector
    "stage_artifacts":      "run",         # Claude, title/ending/non-story
    "stage_wordclass":      "run",         # word class (engine picks how)
    "wordclass_engine":     "spacy",       # "spacy" (free) or "claude" (Haiku)
    "stage_grammar":        "run",         # Claude, SS class on edits
    "stage_sentence":       "run",         # Claude, sentence type + pronoun
    "stage_concern1":       "run",         # Claude, welfare (read its docs)

    # path to the word-difficulty lexicon; set to "" to skip the join
    "difficulty_lexicon":   "word_difficulty_lexicon.csv",

    "model_correct":   "claude-opus-4-7",
    "model_artifacts": "claude-opus-4-7",
    "model_wordclass": "claude-haiku-4-5-20251001",
    "model_grammar":   "claude-sonnet-4-6",
    "model_sentence":  "claude-sonnet-4-6",
    "model_concern1":  "claude-opus-4-7",

    # concurrency: how many scripts to process in parallel on per-script
    # API stages. 8 is a safe default; raise if your rate limit allows,
    # lower if you start seeing 429 errors. Set to 1 for sequential.
    "workers": 8,
}


# ----------------------------------------------------------------------
# Word-difficulty lexicon join (local, free, no API)
# Adds SpellDifficulty + SpellDifficultyConfidence to the word map.
# Only word/contraction tokens are looked up; everything else gets "NA".
# The Confidence column travels with the label so a low-confidence guess
# is never mistaken for a trusted one.
# ----------------------------------------------------------------------
def _join_difficulty(wm, lexicon_path):
    if not lexicon_path or not os.path.exists(lexicon_path):
        return wm
    lex = pd.read_csv(lexicon_path, keep_default_na=False, low_memory=False)

    # Support both old format (DifficultyCategory + Confidence) and
    # the revised format (Revised Category, with Confidence derived from Changed).
    if "DifficultyCategory" in lex.columns:
        cat_col = "DifficultyCategory"
        exclude_vals = {"", "classification"}
    elif "Revised Category" in lex.columns:
        cat_col = "Revised Category"
        exclude_vals = {"", "classification"}   # "word" row is a metadata artifact
    else:
        print("  [difficulty] unrecognised lexicon format — skipping join")
        return wm

    lex = lex[~lex[cat_col].isin(exclude_vals)]

    if "Confidence" in lex.columns:
        conf_col = "Confidence"
    else:
        # Derive from Changed: no → trusted, YES → reviewed, NEW → inferred
        _cmap = {"no": "trusted", "YES": "reviewed", "NEW": "inferred"}
        lex = lex.copy()
        lex["_conf"] = lex["Changed"].map(_cmap).fillna("inferred")
        conf_col = "_conf"

    lex["_key"] = lex["Word"].str.lower().str.strip()
    lex = lex.drop_duplicates("_key", keep="first")
    cat_map  = lex.set_index("_key")[cat_col]
    conf_map = lex.set_index("_key")[conf_col]

    df = wm.copy()
    df["SpellDifficulty"] = "NA"
    df["SpellDifficultyConfidence"] = "NA"
    wordish = df["TokenCategory"].isin(["word", "contraction"])
    keys = (df.loc[wordish, "corr_token"]
            .fillna("").astype(str)
            .str.lower()
            .str.replace(r"[^a-z]", "", regex=True))
    df.loc[wordish, "SpellDifficulty"] = keys.map(cat_map).fillna("NA").values
    # conf_map may be numeric (v7 float) or string; normalise to str so the
    # column dtype stays consistent (prevents pandas TypeError on assignment).
    conf_vals = keys.map(conf_map).fillna("NA").astype(str).replace("nan", "NA")
    df.loc[wordish, "SpellDifficultyConfidence"] = conf_vals.values
    return df


def main(cfg=CONFIG):
    import mech_v2 as m
    import engine_prototype as ep
    import concern1_detector as c1

    t0 = _time.time()
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # keep_default_na=False preserves the deliberate "NA" sentinel so it is
    # never silently converted to NaN on reload (BUILD_BRIEF issue #1).
    df = pd.read_csv(cfg["input_csv"], keep_default_na=False)
    print(f"Loaded {len(df)} scripts from {cfg['input_csv']}")

    # --- Stage 1: correction (produces the corrected-text column) ---
    if cfg["corr_col"] not in df.columns and cfg["stage_correct"] != "skip":
        from corrector_claude import make_claude_corrector
        _set_stage("correction")
        corrector = make_claude_corrector(model=cfg["model_correct"],
                                          api_key=cfg["api_key"])
        if cfg["stage_correct"] == "mock":
            print("MOCK correct: would correct",
                  len(df), "scripts with", cfg["model_correct"])
            df[cfg["corr_col"]] = df[cfg["raw_col"]]      # placeholder
        else:
            from concurrency import parallel_map
            n_workers = cfg.get("workers", 8)
            print(f"Correcting {len(df)} scripts with {n_workers} concurrent workers...")
            results = parallel_map(
                df[cfg["raw_col"]].astype(str).tolist(),
                corrector,
                workers=n_workers,
                label="correct",
            )
            df[cfg["corr_col"]] = [
                r if not isinstance(r, Exception) else "" for r in results
            ]
            n_failed = sum(1 for r in results if isinstance(r, Exception))
            if n_failed:
                print(f"  [warn] {n_failed} scripts failed to correct; "
                      f"left empty in {cfg['corr_col']}")
            print("Correction done.")

            # Change 8 — empty-correction guard.
            # The corrector may catch its own exceptions internally and
            # return "" rather than raising. That slips past the
            # exception check above and produces empty corrected text
            # for affected scripts, which downstream layers process as
            # an all-delete diff (producing a wall of CUTOFF tokens
            # and zero word count). Detect and retry once.
            def _is_suspect(raw, corr, min_ratio=0.3):
                """Flag corrections whose word count drops below
                min_ratio of the raw word count. Ratio-only (no
                absolute floor) so that legitimately short scripts
                are not false-positive flagged."""
                raw_w = len(str(raw).split())
                if raw_w == 0:
                    return False
                return len(str(corr).split()) < min_ratio * raw_w

            suspect_idx = [i for i in range(len(df))
                           if _is_suspect(df.iloc[i][cfg["raw_col"]],
                                          df.iloc[i][cfg["corr_col"]])]
            if suspect_idx:
                ids = df.iloc[suspect_idx][cfg["id_col"]].tolist()
                print(f"  [warn] {len(suspect_idx)} corrections look suspect "
                      f"(empty or <30% of raw word count): {ids}")
                print(f"         retrying these sequentially...")
                col_pos = df.columns.get_loc(cfg["corr_col"])
                still_bad = []
                for i in suspect_idx:
                    raw = df.iloc[i][cfg["raw_col"]]
                    try:
                        new_corr = corrector(str(raw))
                    except Exception as e:
                        new_corr = ""
                        print(f"    [retry-fail] {df.iloc[i][cfg['id_col']]}: "
                              f"{type(e).__name__}: {e}")
                    if not _is_suspect(raw, new_corr):
                        df.iat[i, col_pos] = new_corr
                        print(f"    [ok] {df.iloc[i][cfg['id_col']]}: "
                              f"recovered ({len(str(new_corr).split())} words)")
                    else:
                        still_bad.append(df.iloc[i][cfg["id_col"]])
                if still_bad:
                    print(f"  [ERROR] {len(still_bad)} scripts still produced "
                          f"empty/short corrected text after retry: {still_bad}")
                    print(f"          These will pass through the rest of the "
                          f"pipeline with empty corrected text, producing "
                          f"misleading output (wall of delete rows, CUTOFF "
                          f"tokens, zero word count). Investigate manually.")

        _stage_end("correction")

    # --- Stage 2: title / ending / non-story detection ---
    if cfg["stage_artifacts"] in ("run", "mock"):
        import title_detector as td
        _set_stage("artifacts")
        df = td.run_artifacts(df, id_col=cfg["id_col"],
                              raw_col=cfg["raw_col"],
                              model=cfg["model_artifacts"],
                              api_key=cfg["api_key"],
                              mock=(cfg["stage_artifacts"] == "mock"))
        _stage_end("artifacts")
        print("Artifact stage done"
              + (" (MOCK)" if cfg["stage_artifacts"] == "mock" else ""))

    # --- Stage 3: fixed word layer + 12 cheap columns (always) ---
    _stage_start("word_layer")
    wm = m.run_layer(df, id_col=cfg["id_col"], raw_col=cfg["raw_col"],
                     corr_col=cfg["corr_col"])
    _stage_end("word_layer")
    print(f"Word layer: {len(wm)} token rows.")

    # --- Section 8: boundary-retained text (local, free, no API) ---
    _stage_start("boundary_text")
    br_texts = m.build_boundary_retained_texts(
        wm, texts=df, corr_col=cfg["corr_col"], id_col=cfg["id_col"])
    df["Corrected text (boundaries retained)"] = (
        df[cfg["id_col"]].astype(str).map(br_texts))
    _stage_end("boundary_text")
    print(f"Boundary-retained text built for {len(br_texts)} scripts.")

    # --- Stage 4: word class (engine selectable) ---
    _wc_engine = cfg.get("wordclass_engine", "spacy")
    if cfg["stage_wordclass"] == "run":
        if _wc_engine == "claude":
            _set_stage("wordclass_claude")
            wm = ep.fill_wordclass_with_claude(
                wm, texts=df, model=cfg["model_wordclass"],
                api_key=cfg["api_key"],
                corr_col=cfg["corr_col"], id_col=cfg["id_col"])
            _stage_end("wordclass_claude")
            print(f"Word class filled (Claude, {cfg['model_wordclass']}).")
        else:
            _stage_start("wordclass_spacy")
            wm = ep.fill_with_spacy(wm)
            _stage_end("wordclass_spacy")
            print("Word class filled (spaCy, local).")
    elif cfg["stage_wordclass"] == "mock":
        if _wc_engine == "claude":
            ep.fill_wordclass_with_claude(wm, texts=df, mock=True)
        else:
            ep.fill_with_spacy(wm, mock=True)

    # --- Stage 5: grammar subtype (Claude, only the edits) ---
    if cfg["stage_grammar"] == "run":
        _set_stage("grammar")
        wm = ep.fill_with_claude(wm, model=cfg["model_grammar"],
                                 api_key=cfg["api_key"])
        _stage_end("grammar")
        print("Grammar subtype filled.")
    elif cfg["stage_grammar"] == "mock":
        ep.fill_with_claude(wm, mock=True)

    # --- Stage 5b: word-difficulty lexicon join (local, free) ---
    if cfg.get("difficulty_lexicon"):
        _stage_start("lexicon_join")
        wm = _join_difficulty(wm, cfg["difficulty_lexicon"])
        _stage_end("lexicon_join")
        n_matched = (wm["SpellDifficulty"] != "NA").sum()
        print(f"Word difficulty joined ({n_matched} tokens matched).")

    # --- Stage 6: sentence layer (uses the engine-enriched word map) ---
    _stage_start("sentence_layer")
    sent = m.run_sentence_layer(df, id_col=cfg["id_col"],
                                raw_col=cfg["raw_col"],
                                corr_col=cfg["corr_col"],
                                precomputed_wm=wm)
    _stage_end("sentence_layer")
    print(f"Sentence layer: {len(sent)} sentence rows.")

    # --- Stage 7: sentence type + pronoun reference (Claude) ---
    if cfg["stage_sentence"] == "run":
        _set_stage("sentence")
        sent = ep.fill_sentence_engine(sent, model=cfg["model_sentence"],
                                       api_key=cfg["api_key"])
        _stage_end("sentence")
        print("Sentence type + pronoun reference filled.")
    elif cfg["stage_sentence"] == "mock":
        ep.fill_sentence_engine(sent, mock=True)

    # --- Stage 8: welfare detector (Claude, advisory, isolated) ---
    if cfg["stage_concern1"] in ("run", "mock"):
        _set_stage("concern1")
        df = c1.run_concern1(df, id_col=cfg["id_col"], raw_col=cfg["raw_col"],
                             model=cfg["model_concern1"],
                             api_key=cfg["api_key"],
                             mock=(cfg["stage_concern1"] == "mock"))
        _stage_end("concern1")
        print("Concern 1 stage done"
              + (" (MOCK)" if cfg["stage_concern1"] == "mock" else ""))

    # Patch Concern1 from the final texts df back into sentences so that
    # sentences.csv reflects real values rather than the TBD(engine) placeholder
    # written by classify_scripts inside run_sentence_layer.
    # Concern1 is advisory and never feeds any score; this is display-only.
    if "Concern1" in df.columns:
        _c1_map  = df.set_index(cfg["id_col"])["Concern1"].to_dict()
        _c1r_map = df.set_index(cfg["id_col"]).get("Concern1Reason",
                                                    pd.Series(dtype=str)).to_dict()
        sent["Concern1"] = sent["Identifier"].map(_c1_map).fillna("NA")
        sent["Concern1Reason"] = sent["Identifier"].map(_c1r_map).fillna("NA")

    # --- Save ---
    o = cfg["out_dir"]
    wm.drop(columns=["_ap"], errors="ignore").to_csv(f"{o}/word_map.csv", index=False)
    sent.to_csv(f"{o}/sentences.csv", index=False)
    df.to_csv(f"{o}/texts.csv", index=False)
    print(f"\nSaved word_map.csv, sentences.csv, texts.csv to {o}")
    print("Reminder: Concern1 is advisory and must be reviewed by a human; "
          "it never feeds any score.")

    # --- Token + time summaries ---
    token_summary(out_dir=o)
    time_summary(t0, out_dir=o)


if __name__ == "__main__":
    main()
