"""
run_all.py
==========
One script that runs the whole pipeline in the correct order.

It expects these files sitting next to it, with these exact names:
    aes_canonical.py        (your original, unchanged)
    corrector_claude.py     (your original corrector)
    mech_v2.py
    engine_prototype.py
    concern1_detector.py
    title_detector.py

Edit the CONFIG block (and PRESET below it), then run:  python3 run_all.py

PRESET QUICK REFERENCE
-----------------------
  "all"          — every LLM stage on                         (~$12.40/run)
  "no_correct"   — skip corrector, run all other LLM stages   (~$5.95/run)
  "tb2_only"     — only TB2 rubric assessment                  (~$0.87/run)
  "grammar_sent" — grammar + sentence engine only              (~$2.00/run)
  "local_only"   — zero API calls, all local/free stages       ($0.00/run)

Stage codes (shown in cost summary):
  CORRECT  (C1) — LLM corrector              Opus  4-7   ~$6.45/run
  ARTIFACT (C2) — title / artifact detection  Opus  4-7   ~$1.49/run
  GRAMMAR  (C3) — SS grammar subtype          Sonnet 4-6  ~$0.55/run
  SENTENCE (C4) — sentence type + pronoun     Sonnet 4-6  ~$1.42/run
  WELFARE  (C5) — welfare / Concern 1         Opus  4-7   ~$1.62/run
  TB2      (TB2)— rubric assessment           Sonnet 4-6  ~$0.87/run (cached)
"""

import os
import re
import pandas as pd

# ----------------------------------------------------------------------
# PRESET — change this one line to switch mode
# Options: "all"  "no_correct"  "tb2_only"  "grammar_sent"  "local_only"
# ----------------------------------------------------------------------
PRESET = "all"

# ----------------------------------------------------------------------
# PRESETS definition — each entry overrides the matching CONFIG keys
# ----------------------------------------------------------------------
_PRESETS = {
    # Every LLM stage on. Use for production / full runs.
    "all": {
        "stage_correct":   "run",
        "stage_artifacts":  "run",
        "stage_grammar":   "run",
        "stage_sentence_types":  "run",
        "stage_welfare":   "run",
        "stage_spelling_llm":     "run",
        "stage_breaks":  "run",
        "stage_authorial":   True,
    },
    # Skip corrector (reuse cached column), run all other LLM stages.
    # Use when you've already corrected and just want to re-run analytics.
    "no_correct": {
        "stage_correct":   "skip",
        "stage_artifacts":  "run",
        "stage_grammar":   "run",
        "stage_sentence_types":  "run",
        "stage_welfare":   "run",
        "stage_spelling_llm":     "run",
        "stage_breaks":  "run",
        "stage_authorial":   True,
    },
    # Only TB2 rubric assessment — all other LLM stages skipped.
    # Use when tuning the rubric prompt.
    "tb2_only": {
        "stage_correct":   "skip",
        "stage_artifacts":  "skip",
        "stage_grammar":   "skip",
        "stage_sentence_types":  "skip",
        "stage_welfare":   "skip",
        "stage_spelling_llm":     "skip",
        "stage_breaks":  "skip",
        "stage_authorial":   True,
    },
    # Grammar + sentence engine only — skip corrector, welfare, TB2.
    # Use when tuning grammar or sentence scoring.
    "grammar_sent": {
        "stage_correct":   "skip",
        "stage_artifacts":  "skip",
        "stage_grammar":   "run",
        "stage_sentence_types":  "run",
        "stage_welfare":   "skip",
        "stage_spelling_llm":     "skip",
        "stage_breaks":  "skip",
        "stage_authorial":   False,
    },
    # Zero API calls — only local/free stages (spaCy, lexicon joins, SP/VP/H).
    # Use for rapid iteration on local pipeline logic.
    "local_only": {
        "stage_correct":   "skip",
        "stage_artifacts":  "skip",
        "stage_grammar":   "skip",
        "stage_sentence_types":  "skip",
        "stage_welfare":   "skip",
        "stage_spelling_llm":     "skip",
        "stage_breaks":  "skip",
        "stage_authorial":   False,
    },
}

# ----------------------------------------------------------------------
# CONFIG  — set paths, models, and worker count here.
# Stage on/off is controlled by PRESET above; individual overrides below.
# ----------------------------------------------------------------------
CONFIG = {
    "input_csv":  "gold_clean_365_input.csv",
    "id_col":     "Research ID",
    "raw_col":    "Raw text",
    "corr_col":   "Corrected text (8)",    # created by C1 if not present
    "out_dir":    "./aes_output_clean365",

    "api_key":    os.environ.get("ANTHROPIC_API_KEY", ""),

    # Boundary detection architecture (CB5):
    #   "corrector" — production: corrector terminals + sacred-terminal guard
    #                 (CB5). Rule 2/3/BR1 detectors are OFF.
    #   "rules"     — legacy: rule-based detectors (Rule 1/2/3 + BR1).
    "boundary_mode": "corrector",

    # --- Stage toggles (set via PRESET above; override here if needed) ---
    # C1  CORRECT  — LLM corrector (Opus)          "run" / "mock" / "skip"
    "stage_correct":   "run",
    # C2  ARTIFACT — title / artifact detection (Opus) "run" / "mock" / "skip"
    "stage_artifacts":  "run",
    # C3  GRAMMAR  — SS grammar subtype (Sonnet)   "run" / "mock" / "skip"
    "stage_grammar":   "run",
    # C4  SENTENCE — sentence type + pronoun (Sonnet) "run" / "mock" / "skip"
    "stage_sentence_types":  "run",
    # C5  WELFARE  — Concern 1 welfare flag (Opus)  "run" / "mock" / "skip"
    "stage_welfare":   "run",
    # SF  SPELL-FINAL Gate 2 (Sonnet) "run" / "mock" / "skip"
    # SF base label + Gate 1 always run (deterministic). Gate 2 needs an
    # api_key; on "skip" or local-only runs, remaining Simple/Common
    # misspellings stay as Incorrect (conservative default).
    "stage_spelling_llm":     "run",
    # BK2-missing  — missing-break LLM stage (Sonnet) "run" / "skip"
    # Runs after BK2 deterministic; only upgrades na → missing. Gated on
    # api_key. Skipped in local_only / tb2_only / grammar_sent presets.
    "stage_breaks":  "run",
    # TB2 RUBRIC   — text-level rubric (Sonnet)     True / False
    "stage_authorial":   True,
    # CM  CLAUSE-MAP — optional fine-grained clause/phrase layer (Sonnet)
    # OFF by default. Set to "run" to produce clause_map.csv. Needs api_key.
    # Explanatory / teacher-facing only; not wired into scoring. Adds a few
    # dollars to a full run. Requires C4 (Sentence type) to have run.
    "stage_clauses": "skip",

    # word-class engine (always runs; free with spaCy)
    "wordclass_engine":     "spacy",       # "spacy" (free) or "claude" (Haiku)

    # lexicon paths (set to "" to skip that join)
    # Single source of truth: lexicon_master.csv carries both the difficulty
    # column (Final Word Category) and the vocab columns (VocabCategory_LLM,
    # VocabConfidence, VocabCategory, VocabSimplerAlt). It replaces the three
    # near-duplicate files (Spelling_FINAL_v20_tidy / word_difficulty_lexicon /
    # Spelling_with_VocabClass_LLM), which were the same 102,716 words.
    "difficulty_lexicon":   "lexicon_master.csv",
    "lexicon_dir":          "lexicons",
    "vocab_lexicon":        "lexicon_master.csv",

    # models
    "model_C1_correct":   "claude-opus-4-7",
    "model_C2_artifact":  "claude-opus-4-7",
    "model_C3_grammar":   "claude-sonnet-4-6",
    "model_C4_sentence":  "claude-sonnet-4-6",
    "model_C5_welfare":   "claude-opus-4-7",
    "model_SF_gate2":     "claude-sonnet-4-6",
    "model_BK2_missing":  "claude-sonnet-4-6",
    "model_TB2_rubric":   "claude-sonnet-4-6",

    # concurrency
    "workers": 8,

    # TB2 prompt fallback (per-script Prompt column takes precedence)
    "prompt_text": "",

    # B2 PROMPT artifact detection (local, Levenshtein)
    "tag_prompt_copies": True,
}


def _apply_preset(cfg, preset_name):
    """Apply a named preset, then print a summary of active stages."""
    if preset_name not in _PRESETS:
        print(f"  [preset] Unknown preset {preset_name!r}; ignoring. "
              f"Available: {list(_PRESETS)}")
        return cfg
    cfg = dict(cfg)
    cfg.update(_PRESETS[preset_name])

    # Backward-compatibility shim. The stage toggles were renamed from the old
    # C1..C5 / SF / BK2 / TB2 / CM codes to logical names. Any config written
    # against an old key is mirrored to the new name (and vice-versa) so
    # external presets or overrides using the old codes keep working.
    _STAGE_ALIASES = {
        "stage_C1_correct":    "stage_correct",
        "stage_C2_artifact":   "stage_artifacts",
        "stage_C3_grammar":    "stage_grammar",
        "stage_C4_sentence":   "stage_sentence_types",
        "stage_C5_welfare":    "stage_welfare",
        "stage_SF_gate2":      "stage_spelling_llm",
        "stage_BK2_missing":   "stage_breaks",
        "stage_TB2_rubric":    "stage_authorial",
        "stage_CM_clausemap":  "stage_clauses",
    }
    for _old, _new in _STAGE_ALIASES.items():
        if _old in cfg:
            cfg.setdefault(_new, cfg[_old])
        if _new in cfg:
            cfg.setdefault(_old, cfg[_new])

    # Build cost-estimate string
    _est = {
        "all": "~$12.40", "no_correct": "~$5.95", "tb2_only": "~$0.87",
        "grammar_sent": "~$2.00", "local_only": "$0.00",
    }
    _on = []
    for label, key in [("CORRECT", "stage_correct"),
                       ("ARTIFACTS", "stage_artifacts"),
                       ("GRAMMAR", "stage_grammar"),
                       ("SENTENCE-TYPES", "stage_sentence_types"),
                       ("WELFARE", "stage_welfare"),
                       ("SPELLING-LLM", "stage_spelling_llm"),
                       ("BREAKS", "stage_breaks"),
                       ("CLAUSES", "stage_clauses")]:
        v = cfg.get(key, "skip")
        _on.append(f"{'✓' if v == 'run' else '✗'} {label}")
    _on.append(f"{'✓' if cfg.get('stage_authorial') else '✗'} AUTHORIAL")
    print(f"\n  PRESET={preset_name!r}  {_est.get(preset_name, '')}")
    print("  " + "  ".join(_on) + "\n")
    return cfg


# ----------------------------------------------------------------------
# Word-difficulty lexicon join (local, free, no API)
# Adds SpellDifficulty + SpellDifficultyConfidence to the word map.
# Only word/contraction tokens are looked up; everything else gets "NA".
# The Confidence column travels with the label so a low-confidence guess
# is never mistaken for a trusted one.
# ----------------------------------------------------------------------
def _join_difficulty(wm, lexicon_path):
    """
    SP1 (Phase SP): Populate WordKind, SpellDifficulty, SpellDifficultyConfidence,
    and DualSpelling from the reconciled lexicon.

    Both WordKind AND SpellDifficulty now come from the same column
    (`Revised Category` / Reconciled Category), so they agree 100%.
    Previously SpellDifficulty used `Final Word Category` (a cleaned-down
    four-tier signal that dropped proper_noun_abbreviation and junk info),
    while WordKind used `Revised Category`. The source switch eliminates
    all drift between the two columns.

    The four-tier algorithm counts (Simple/Common/Difficult/Challenging)
    should use is_spelling_assessable() to exclude proper_noun_abbreviation,
    junk, and NA tokens before counting.
    """
    if not lexicon_path or not os.path.exists(lexicon_path):
        return wm
    lex = pd.read_csv(lexicon_path, keep_default_na=False, low_memory=False)

    # SP1 — prefer Revised Category (the Reconciled Category) for both
    # WordKind and SpellDifficulty.  Fall back to legacy formats if the
    # column is absent so existing test data still works.
    if "Revised Category" in lex.columns:
        cat_col = "Revised Category"  # single source of truth for both columns
        exclude_vals = {"", "classification"}
    elif "Final Word Category" in lex.columns:
        cat_col = "Final Word Category"  # lexicon_master.csv format
        exclude_vals = {"", "classification"}
    elif "DifficultyCategory" in lex.columns:
        cat_col = "DifficultyCategory"
        exclude_vals = {"", "classification"}
    else:
        print("  [difficulty] unrecognised lexicon format — skipping join")
        return wm, {}

    lex = lex[~lex[cat_col].isin(exclude_vals)]

    if "Confidence" in lex.columns:
        conf_col = "Confidence"
    else:
        # Derive from Changed: no → trusted, YES → reviewed, NEW → inferred
        _cmap = {"no": "trusted", "YES": "reviewed", "NEW": "inferred"}
        lex = lex.copy()
        lex["_conf"] = lex["Changed"].map(_cmap).fillna("inferred")
        conf_col = "_conf"

    lex["_key"] = lex["Word"].astype(str).str.lower().str.strip()
    lex = lex.drop_duplicates("_key", keep="first")
    cat_map  = lex.set_index("_key")[cat_col]
    conf_map = lex.set_index("_key")[conf_col]

    # SP1 — DualSpelling lookup dict (word_lower → bool)
    if "Dual Spelling" in lex.columns:
        dual_series = lex.set_index("_key")["Dual Spelling"]
        dual_map = {k: (str(v).upper() == "TRUE")
                    for k, v in dual_series.items()}
    else:
        dual_map = {}

    df = wm.copy()
    df["SpellDifficulty"] = "NA"
    df["SpellDifficultyConfidence"] = "NA"
    df["WordKind"] = "NA"
    df["DualSpelling"] = "FALSE"   # NEW column (SP1)

    wordish = df["TokenCategory"].isin(["word", "contraction"])
    keys = (df.loc[wordish, "corr_token"]
            .fillna("").astype(str)
            .str.lower()
            .str.replace(r"[^a-z]", "", regex=True))

    # Both SpellDifficulty and WordKind get the same Reconciled Category value
    df.loc[wordish, "SpellDifficulty"] = keys.map(cat_map).fillna("NA").astype(str).values
    df.loc[wordish, "SpellDifficultyConfidence"] = keys.map(conf_map).fillna("NA").astype(str).values
    df.loc[wordish, "WordKind"] = keys.map(cat_map).fillna("NA").astype(str).values
    df.loc[wordish, "DualSpelling"] = keys.map(
        lambda k: "TRUE" if dual_map.get(k, False) else "FALSE"
    ).values

    return df, dual_map


def _apply_dual_spelling_fix(wm, dual_map):
    """
    SP2 (Phase SP): Honour the Dual Spelling flag.

    When the corrector substituted one valid regional spelling for another
    (e.g. meters → metres, color → colour), the student should NOT be
    marked Incorrect.  If both raw_token and corr_token are in the lexicon
    and at least one carries DualSpelling=TRUE, flip Spell class from
    Incorrect → Correct and record DualSpellingResolved=TRUE.
    """
    df = wm.copy()
    df["DualSpellingResolved"] = "FALSE"   # NEW column (SP2), default FALSE

    if not dual_map:
        return df

    # Only look at rows already flagged Incorrect (safe, conservative override)
    incorrect_mask = (
        df.get("Spell class", pd.Series(dtype=str)) == "Incorrect"
    )
    if not incorrect_mask.any():
        return df

    n_flipped = 0
    for idx in df[incorrect_mask].index:
        raw = str(df.at[idx, "raw_token"] or "").lower().strip()
        corr = str(df.at[idx, "corr_token"] or "").lower().strip()

        if not raw or not corr or raw == corr:
            continue

        # Strip non-alpha for consistent lookup
        raw_key  = re.sub(r"[^a-z]", "", raw)
        corr_key = re.sub(r"[^a-z]", "", corr)

        if not raw_key or not corr_key:
            continue

        # Both must be known words; at least one must be flagged DualSpelling
        raw_known  = raw_key  in dual_map
        corr_known = corr_key in dual_map
        if not (raw_known and corr_known):
            continue
        if not (dual_map.get(raw_key, False) or dual_map.get(corr_key, False)):
            continue

        # Override: this is a valid alternate spelling, not an error
        df.at[idx, "Spell error"]           = "FALSE"
        df.at[idx, "Spell subtype"]         = "Correct"
        df.at[idx, "Spell class"]           = "Correct"
        df.at[idx, "DualSpellingResolved"]  = "TRUE"
        n_flipped += 1

    if n_flipped:
        print(f"  [SP2] {n_flipped} dual-spelling Incorrect→Correct override(s) applied.")
    return df


def is_spelling_assessable(row):
    """
    SP3 (Phase SP): Determine whether a word token should be included in
    any spelling tier count (Simple, Common, Difficult, Challenging).

    Excluded:
    - Non-word tokens (TokenCategory != 'word')
    - Proper nouns (Spell class = 'ProperNoun-ignored', or
                    WordKind = 'proper_noun_abbreviation')
    - Junk tokens (WordKind = 'junk' or 'name_foreign')
    - Delete-op rows (op = 'delete')
    - Tokens without a tier (WordKind = 'NA')
    """
    if str(row.get("TokenCategory", "")) != "word":
        return False
    if str(row.get("Spell class", "")) == "ProperNoun-ignored":
        return False
    if str(row.get("WordKind", "")) in ("proper_noun_abbreviation", "junk",
                                         "name_foreign", "NA"):
        return False
    if str(row.get("op", "")) == "delete":
        return False
    return True


def is_spell_error(row):
    """
    SP3: Determine whether a word token counts as a spell error.

    True for tokens that are spelling-assessable AND have Spell error = TRUE.
    """
    if not is_spelling_assessable(row):
        return False
    return str(row.get("Spell error", "")).upper() == "TRUE"


def _cp_dir(cfg):
    return os.path.join(cfg["out_dir"], "checkpoints")

def _cp_save(tag, cfg, df=None, wm=None, sent=None):
    """Write a checkpoint snapshot after a paid LLM stage."""
    d = _cp_dir(cfg)
    os.makedirs(d, exist_ok=True)
    if df   is not None: df.to_csv(  os.path.join(d, f"{tag}_texts.csv"),    index=False)
    if wm   is not None: wm.to_csv(  os.path.join(d, f"{tag}_word_map.csv"), index=False)
    if sent is not None: sent.to_csv(os.path.join(d, f"{tag}_sentences.csv"),index=False)
    print(f"  [CP] checkpoint {tag!r} saved to {d}/")

def _cp_load(tag, cfg):
    """Return (df, wm, sent) from a checkpoint, any absent frame is None."""
    d = _cp_dir(cfg)
    def _read(fname):
        p = os.path.join(d, fname)
        return pd.read_csv(p, keep_default_na=False, low_memory=False) if os.path.exists(p) else None
    return _read(f"{tag}_texts.csv"), _read(f"{tag}_word_map.csv"), _read(f"{tag}_sentences.csv")

def _cp_exists(tag, cfg):
    d = _cp_dir(cfg)
    return any(os.path.exists(os.path.join(d, f"{tag}_{f}.csv"))
               for f in ("texts", "word_map", "sentences"))

def main(cfg=CONFIG, preset=None):
    import mech_v2 as m
    import engine_prototype as ep
    import concern1_detector as c1

    # Apply preset (module-level PRESET if not passed explicitly)
    cfg = _apply_preset(cfg, preset or PRESET)

    os.makedirs(cfg["out_dir"], exist_ok=True)

    # keep_default_na=False preserves the deliberate "NA" sentinel so it is
    # never silently converted to NaN on reload (BUILD_BRIEF issue #1).
    df = pd.read_csv(cfg["input_csv"], keep_default_na=False)
    print(f"Loaded {len(df)} scripts from {cfg['input_csv']}")

    # --- Checkpoint resume: if a checkpoint exists, reload and fast-forward ---
    # Check from latest to earliest; the first hit wins.
    _cp_resume = None
    for _tag in ("cp6_authorial", "cp5_welfare", "cp4_sentence", "cp3_grammar", "cp2_artifact", "cp1_correct"):
        if _cp_exists(_tag, cfg):
            _cp_resume = _tag
            break
    # --- Checkpoint resume: frame-aware. Each checkpoint saved only the
    # frame(s) that changed, so we load each frame from its own deepest
    # checkpoint, then derive which stages are already done.
    #   df-bearing checkpoints (newest→oldest): cp6 cp5 cp2 cp1
    #   sent-bearing:                            cp4
    #   wm-bearing:                              cp3   (may be absent)
    wm = None
    sent = None
    _done = set()   # stage tags already satisfied by a loaded frame
    def _first_with(frame_key, tags):
        for t in tags:
            d = _cp_dir(cfg)
            if os.path.exists(os.path.join(d, f"{t}_{frame_key}.csv")):
                return t
        return None

    # Partial TB2 checkpoint tags, highest first (e.g. cp6t_230 means 230 scripts done)
    _tb2_partial_tags = tuple(f"cp6t_{n:03d}" for n in range(900, 0, -10))
    _df_cp   = _first_with("texts", _tb2_partial_tags + ("cp6_authorial", "cp5_welfare", "cp2_artifact", "cp1_correct"))
    _wm_cp   = _first_with("word_map",  ("cp3_grammar",))
    _sent_cp = _first_with("sentences", ("cp4_sentence",))

    if _df_cp:
        _df_loaded, _, _ = _cp_load(_df_cp, cfg)
        if _df_loaded is not None:
            df = _df_loaded
            print(f"  [CP] resume: df loaded from {_df_cp!r} ({len(df)} rows)")
            # df checkpoints are cumulative on the texts frame
            _done.add("cp1_correct")
            _is_tb2_partial = _df_cp.startswith("cp6t_")
            if _is_tb2_partial:
                # Partial TB2: C1/C2/C5 all completed before TB2 started
                _done.add("cp2_artifact")
                _done.add("cp5_welfare")
                # cp6_authorial NOT added — TB2 needs to finish
            else:
                if _df_cp in ("cp2_artifact", "cp5_welfare", "cp6_authorial"): _done.add("cp2_artifact")
                if _df_cp in ("cp5_welfare", "cp6_authorial"):                 _done.add("cp5_welfare")
                if _df_cp == "cp6_authorial":                                  _done.add("cp6_authorial")
    if _wm_cp:
        _, _wm_loaded, _ = _cp_load(_wm_cp, cfg)
        if _wm_loaded is not None:
            wm = _wm_loaded
            _done.add("cp3_grammar")
            print(f"  [CP] resume: wm loaded from {_wm_cp!r} ({len(wm)} rows)")
    if _sent_cp:
        _, _, _sent_loaded = _cp_load(_sent_cp, cfg)
        if _sent_loaded is not None:
            sent = _sent_loaded
            _done.add("cp4_sentence")
            print(f"  [CP] resume: sent loaded from {_sent_cp!r} ({len(sent)} rows)")
    if _done:
        print(f"  [CP] stages satisfied by checkpoints: {sorted(_done)}")


    # --- Pa1: HTML + Prompt passthrough ---
    # New input has "HTML" and "Prompt" columns. Map them to pipeline names
    # early so all downstream stages (TB1, Pa scorer) see them.
    # Backward-compatible: if columns absent, add empty strings.
    if "HTML" in df.columns and "Raw HTML" not in df.columns:
        df["Raw HTML"] = df["HTML"].fillna("").astype(str)
    elif "Raw HTML" not in df.columns:
        df["Raw HTML"] = ""
    if "Prompt" not in df.columns:
        df["Prompt"] = ""

    # Resume is driven by the _done set (which stages a loaded frame satisfies).

    # --- Stage 1: correction (produces the corrected-text column) ---
    # When C1 is skipped and corr_col is absent, load from cached texts.csv.
    # If no cache exists either, fall back to raw text (allows grammar/TB2
    # stages to run on uncorrected input with a warning).
    if "cp1_correct" in _done:
        print(f"  [CP] C1 correct skipped — df loaded from checkpoint")
    elif cfg["corr_col"] not in df.columns and cfg["stage_correct"] == "skip":
        cache_path = os.path.join(cfg["out_dir"], "texts.csv")
        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, keep_default_na=False, low_memory=False)
            if cfg["corr_col"] in cached.columns and cfg["id_col"] in cached.columns:
                corr_map = cached.set_index(cfg["id_col"])[cfg["corr_col"]].to_dict()
                df[cfg["corr_col"]] = df[cfg["id_col"]].map(corr_map).fillna(
                    df[cfg["raw_col"]])
                print(f"C1 skipped — loaded '{cfg['corr_col']}' from cached texts.csv")
            else:
                df[cfg["corr_col"]] = df[cfg["raw_col"]]
                print(f"C1 skipped — cache missing column; using raw text as fallback")
        else:
            df[cfg["corr_col"]] = df[cfg["raw_col"]]
            print(f"C1 skipped — no cache; using raw text as fallback")

    if cfg["corr_col"] not in df.columns and cfg["stage_correct"] != "skip":
        from corrector_claude import make_claude_corrector
        corrector = make_claude_corrector(model=cfg["model_C1_correct"],
                                          api_key=cfg["api_key"])
        if cfg["stage_correct"] == "mock":
            print("MOCK correct: would correct",
                  len(df), "scripts with", cfg["model_C1_correct"])
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
            # any exceptions returned in results are kept as the value;
            # the runner is permissive and lets the cheap layer mark them
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

    if "cp1_correct" not in _done and cfg["stage_correct"] == "run" and cfg["corr_col"] in df.columns:
        _cp_save("cp1_correct", cfg, df=df)

    # --- Stage 2: title / ending / non-story detection ---
    if "cp2_artifact" in _done:
        print(f"  [CP] C2 artifact skipped — df loaded from checkpoint")
    elif cfg["stage_artifacts"] in ("run", "mock"):
        import title_detector as td
        df = td.run_artifacts(df, id_col=cfg["id_col"],
                              raw_col=cfg["raw_col"],
                              model=cfg["model_C2_artifact"],
                              api_key=cfg["api_key"],
                              mock=(cfg["stage_artifacts"] == "mock"))
        print("Artifact stage done"
              + (" (MOCK)" if cfg["stage_artifacts"] == "mock" else ""))
        _cp_save("cp2_artifact", cfg, df=df)

    # --- Stage 3: fixed word layer + 12 cheap columns (always) ---
    if wm is None:
        wm = m.run_layer(df, id_col=cfg["id_col"], raw_col=cfg["raw_col"],
                         corr_col=cfg["corr_col"],
                         boundary_mode=cfg.get("boundary_mode", "rules"))
        print(f"Word layer: {len(wm)} token rows. "
              f"[boundary_mode={cfg.get('boundary_mode', 'rules')}]")
    else:
        print(f"Word layer: {len(wm)} token rows. [restored from checkpoint]")

    # --- Section 8: boundary-retained text (local, free, no API) ---
    br_texts = m.build_boundary_retained_texts(
        wm, texts=df, corr_col=cfg["corr_col"], id_col=cfg["id_col"])
    df["Corrected text (boundaries retained)"] = (
        df[cfg["id_col"]].astype(str).map(br_texts))
    print(f"Boundary-retained text built for {len(br_texts)} scripts.")

    # --- Stage 4: word class (engine selectable, always runs) ---
    if "cp3_grammar" in _done:
        print(f"  [CP] word-class/grammar/SF1/VP1 skipped — wm loaded from checkpoint")
    else:
        _wc_engine = cfg.get("wordclass_engine", "spacy")
        if _wc_engine == "claude":
            wm = ep.fill_wordclass_with_claude(
                wm, texts=df, model=cfg.get("model_C1_correct", "claude-opus-4-7"),
                api_key=cfg["api_key"],
                corr_col=cfg["corr_col"], id_col=cfg["id_col"])
            print(f"Word class filled (Claude).")
        else:
            wm = ep.fill_with_spacy(wm)
            print("Word class filled (spaCy, local).")

        # --- Stage 5: grammar subtype (Claude, only the edits) ---
        if cfg["stage_grammar"] == "run":
            wm = ep.fill_with_claude(wm, model=cfg["model_C3_grammar"],
                                     api_key=cfg["api_key"])
            print("Grammar subtype filled.")
        elif cfg["stage_grammar"] == "mock":
            ep.fill_with_claude(wm, mock=True)

    # --- Stage 5b: word-difficulty lexicon join (local, free) ---
    # SP1: _join_difficulty now returns (wm, dual_map) and populates both
    #      WordKind AND SpellDifficulty from the same Reconciled Category.
    # SP2: _apply_dual_spelling_fix honours the Dual Spelling flag.
    if cfg.get("difficulty_lexicon"):
        wm, _dual_map = _join_difficulty(wm, cfg["difficulty_lexicon"])
        n_matched = (wm["SpellDifficulty"] != "NA").sum()
        n_dual = (wm["DualSpelling"] == "TRUE").sum()
        print(f"Word difficulty joined ({n_matched} tokens matched, "
              f"{n_dual} dual-spelling tokens).")
        # SP2 — flip dual-spelling Incorrect → Correct
        wm = _apply_dual_spelling_fix(wm, _dual_map)
        n_wk_sd_agree = (wm["WordKind"] == wm["SpellDifficulty"]).sum()
        n_wk_total = (wm["WordKind"] != "").sum()
        print(f"  [SP1] WordKind/SpellDifficulty agreement: "
              f"{n_wk_sd_agree}/{n_wk_total} tokens "
              f"({100*n_wk_sd_agree/max(n_wk_total,1):.1f}%)")

    # --- Stage 5b-WB1: contraction POS override (after spaCy, before chunker) ---
    wm = m.apply_contraction_pos_override(wm)
    n_contr = (wm["TokenCategory"] == "contraction").sum()
    print(f"Contraction POS override applied ({n_contr} contraction tokens).")

    # --- Stage 5b-SF1: SpellingFinal base label + Gate 1 (deterministic) ---
    # Adds SpellingFinal + SpellingFinalGate columns. Gate 1 catches typos
    # whose corrected form appears spelled correctly elsewhere in the same
    # script. Gate 2 (LLM) is applied later if stage_spelling_llm is "run".
    import spelling_final as sf
    wm = sf.apply_base_and_gate1(wm)
    n_gate1 = int(wm.attrs.get("sf_gate1_count", 0))
    print(f"  [SF1] SpellingFinal labelled; Gate 1 reclassified "
          f"{n_gate1} Simple/Common-Incorrect tokens as Typo.")

    # --- Stage 5b-SF1-Gate2: light LLM classification (optional, API) ---
    if cfg.get("stage_spelling_llm") == "run" and cfg.get("api_key"):
        wm = sf.apply_gate2(wm,
                            model=cfg.get("model_SF_gate2",
                                          "claude-sonnet-4-6"),
                            api_key=cfg["api_key"],
                            mock=False)
        n_typo = int(wm.attrs.get("sf_gate2_typo", 0))
        n_ns = int(wm.attrs.get("sf_gate2_notspelling", 0))
        print(f"  [SF1] Gate 2 LLM: +{n_typo} Typo, +{n_ns} NotSpelling.")
    elif cfg.get("stage_spelling_llm") == "mock":
        wm = sf.apply_gate2(wm, model="", api_key="", mock=True)
        print(f"  [SF1] Gate 2 skipped (mock).")
    else:
        print(f"  [SF1] Gate 2 skipped (no api_key / stage skip); "
              "remaining Simple/Common-Incorrect stay as Incorrect.")

    # --- Stage 5c: vocabulary structure layer (local, free) ---
    # Reads the three text-file lexicons produced by
    # export_lexicon_to_files.py from the LLM-classified
    # Spelling_with_VocabClass_LLM.csv. Adds SFL group structure
    # (GroupID, GroupType, GroupRole), clause depth, lexical/grammatical
    # split, three-tier precision (precise/specific/simple), cohesion
    # role, and UsedAppropriately columns. No API calls, deterministic.
    if cfg.get("lexicon_dir"):
        try:
            import word_map_chunker_v3 as chunker
            precise, specific, simple, phrases = chunker.load_lexicon(
                cfg["lexicon_dir"])
            if precise or specific or simple:
                wm = chunker.chunk_and_classify(
                    wm, precise, specific, simple, phrases)
                n_lex = (wm["LexicalPrecision"].notna()).sum()
                print(f"Vocabulary structure layer done "
                      f"({n_lex} lexical tokens classified; "
                      f"lexicons: precise={len(precise)}, "
                      f"specific={len(specific)}, simple={len(simple)}).")
            else:
                print("  [vocab] lexicon_dir set but no lexicon files found; "
                      "skipping Stage 5c")
        except ImportError:
            print("  [vocab] word_map_chunker_v3 not importable; "
                  "skipping Stage 5c")
        except Exception as e:
            print(f"  [vocab] Stage 5c failed: {e}; continuing without it")

    # --- Stage 5c-VP1: VocabCategory from LLM-classified lexicon (local, free) ---
    # Phase VP: populates VocabCategory (precise/specific/simple/NA) and
    # VocabCategoryConfidence. Must run AFTER Stage 5b so WordKind=
    # proper_noun_abbreviation is already set.
    if cfg.get("vocab_lexicon"):
        try:
            from vocab_classifier import assign_vocab_columns
            wm = assign_vocab_columns(wm, cfg["vocab_lexicon"])
            n_prec = (wm.get("VocabCategory", pd.Series([])) == "precise").sum()
            n_spec = (wm.get("VocabCategory", pd.Series([])) == "specific").sum()
            n_simp = (wm.get("VocabCategory", pd.Series([])) == "simple").sum()
            print(f"VP1 VocabCategory done "
                  f"(precise={n_prec}, specific={n_spec}, simple={n_simp}).")
        except Exception as e:
            print(f"  [VP1] VocabCategory failed: {e}; continuing without it")

    # --- Stage 5d: B2 PROMPT artifact tagging (local, free) ---
    if cfg.get("tag_prompt_copies") and cfg.get("prompt_text"):
        try:
            from title_detector import tag_prompt_copies_in_wm
            wm = tag_prompt_copies_in_wm(wm, cfg["prompt_text"])
            n_prompt = (wm.get("TextualArtifact", pd.Series([])) == "PROMPT").sum()
            print(f"B2 PROMPT artifact tagging done ({n_prompt} rows tagged).")
        except Exception as e:
            print(f"  [B2] PROMPT tagging failed: {e}; continuing without it")

    if "cp3_grammar" not in _done and cfg["stage_grammar"] == "run":
        _cp_save("cp3_grammar", cfg, wm=wm)

    # --- Stage 6: sentence layer (uses the engine-enriched word map) ---
    if "cp4_sentence" in _done:
        print(f"  [CP] sentence layer skipped — sent loaded from checkpoint")
        if sent is None:
            raise RuntimeError("checkpoint claims sent exists but _cp_load returned None")
    else:
        sent = m.run_sentence_layer(df, id_col=cfg["id_col"],
                                    boundary_mode=cfg.get("boundary_mode", "rules"),
                                    raw_col=cfg["raw_col"],
                                    corr_col=cfg["corr_col"],
                                    precomputed_wm=wm)
    print(f"Sentence layer: {len(sent)} sentence rows.")

    # BK1/BK2/BK2-missing only run when sent was freshly built. When sent is
    # restored from the cp4 checkpoint it already carries these columns (cp4
    # is saved AFTER BK2-missing and C4), so re-running them would re-map and,
    # for BK2-missing, double-count and re-spend.
    if "cp4_sentence" in _done:
        print("  [CP] BK1/BK2/BK2-missing skipped — sent loaded from checkpoint")
    else:
        # --- Stage 6b-BK1: HTML break mapping (deterministic, local) ---
        # Adds ParagraphID, BlockType, PrecededByBreak, BlockStyle,
        # BlockPosition + BlockMatchConfidence columns to sentences. Foundation
        # for Step 2 break-type classifications (paragraph/speech/structural).
        try:
            import break_mapping as bk
            sent = bk.map_sentences_to_blocks(sent, df, id_col=cfg["id_col"],
                                              html_col="Raw HTML")
            n_low = int((sent["BlockMatchConfidence"] == "low").sum())
            pct_low = 100 * n_low / max(len(sent), 1)
            print(f"  [BK1] Block mapping done; "
                  f"{n_low}/{len(sent)} sentences flagged low-confidence "
                  f"({pct_low:.1f}%).")
        except Exception as e:
            print(f"  [BK1] failed: {e}; continuing without break mapping")

        # --- Stage 6c-BK2: break-correctness classifications (deterministic) ---
        # ParagraphBreakClass / SpeechBreakClass / StructuralBreak /
        # TimeSlipStatus columns, plus TIMESLIP TextualArtifact for isolated
        # temporal fragments. The "missing paragraph break" judgement from the
        # corrector pipeline is not run here -- inline temporal fragments still
        # produce ParagraphBreakClass='missing' via the option-(ii) rule.
        try:
            import break_classifications as bc
            sent = bc.apply_all(sent, wm)
            n_conf = int(sent.attrs.get("bk2_confirmed_timeslips", 0))
            n_inl = int(sent.attrs.get("bk2_inline_timeslips", 0))
            n_para_true = int((sent["ParagraphBreakClass"] == "true").sum())
            n_para_sup = int((sent["ParagraphBreakClass"] == "superfluous").sum())
            n_speech = int((sent["SpeechBreakClass"] != "na").sum())
            print(f"  [BK2] Paragraph: {n_para_true} true, "
                  f"{n_para_sup} superfluous. "
                  f"Speech-change events: {n_speech}. "
                  f"TIMESLIP: {n_conf} confirmed, {n_inl} inline-missing.")
        except Exception as e:
            print(f"  [BK2] failed: {e}; continuing without break classifications")

        # --- Stage 6c-BK2-final: missing-break LLM stage (optional, API) ---
        # Runs after BK2 deterministic; only upgrades na → missing. One call per
        # script (~$0.01-0.02). Skipped when no api_key or stage_breaks=skip.
        if cfg.get("stage_breaks") == "run" and cfg.get("api_key"):
            try:
                import break_classifications as bc
                n_missing_before = int((sent["ParagraphBreakClass"] == "missing").sum())
                sent = bc.apply_missing_break_llm(
                    sent,
                    model=cfg.get("model_BK2_missing", "claude-sonnet-4-6"),
                    api_key=cfg["api_key"],
                )
                n_missing_after = int((sent["ParagraphBreakClass"] == "missing").sum())
                n_added = n_missing_after - n_missing_before
                n_scripts = sent["Identifier"].nunique()
                avg = n_added / max(n_scripts, 1)
                print(f"  [BK2-missing] LLM stage done; "
                      f"+{n_added} missing breaks added "
                      f"({avg:.2f}/script avg). "
                      f"Total missing: {n_missing_after}.")
                if avg > 2.5:
                    print(f"  [BK2-missing] WARN: avg {avg:.2f}/script exceeds "
                          f"conservatism target (~1-2); consider tightening prompt.")
            except Exception as e:
                print(f"  [BK2-missing] failed: {e}; continuing without LLM missing breaks")
        else:
            reason = "no api_key" if not cfg.get("api_key") else "stage skip"
            print(f"  [BK2-missing] skipped ({reason}); deterministic missing breaks only.")

    # --- Stage 7: sentence type + pronoun reference (Claude) ---
    if "cp4_sentence" in _done:
        print(f"  [CP] C4 sentence-engine skipped — sent loaded from checkpoint")
    else:
        if cfg["stage_sentence_types"] == "run":
            sent = ep.fill_sentence_engine(sent, model=cfg["model_C4_sentence"],
                                           api_key=cfg["api_key"])
            print("Sentence type + pronoun reference filled.")
        elif cfg["stage_sentence_types"] == "mock":
            ep.fill_sentence_engine(sent, mock=True)
        if cfg["stage_sentence_types"] == "run":
            _cp_save("cp4_sentence", cfg, sent=sent)

    # --- Stage 7b: clause map (OPTIONAL add-on, off by default) ---
    if cfg.get("stage_clauses") == "run" and cfg.get("api_key"):
        import clause_map as cm
        clause_df = cm.build_clause_map(sent, model=cfg.get("model_C4_sentence"),
                                        api_key=cfg["api_key"])
        clause_df.to_csv(f"{cfg['out_dir']}/clause_map.csv", index=False)
        print(f"Clause map built: {len(clause_df)} units "
              f"-> {cfg['out_dir']}/clause_map.csv")
    elif cfg.get("stage_clauses") == "run":
        print("  [clause_map] skipped: needs an api_key.")

    # --- Stage 8: welfare detector (Claude, advisory, isolated) ---
    if "cp5_welfare" in _done:
        print(f"  [CP] C5 welfare skipped — df loaded from checkpoint")
    else:
        if cfg["stage_welfare"] in ("run", "mock"):
            df = c1.run_concern1(df, id_col=cfg["id_col"], raw_col=cfg["raw_col"],
                                 model=cfg["model_C5_welfare"],
                                 api_key=cfg["api_key"],
                                 mock=(cfg["stage_welfare"] == "mock"))
            print("Concern 1 stage done"
                  + (" (MOCK)" if cfg["stage_welfare"] == "mock" else ""))
            _cp_save("cp5_welfare", cfg, df=df)

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

    # Change 12 — "Title or heading only" rubric trigger.
    # A script qualifies when the title detector found a title and the
    # narrative body (corrected words minus title, ending, and other
    # artifact words) is at most 3 words. Per the rubric, such scripts
    # score AU=1 and every other criterion 0; downstream scoring reads
    # ScriptClass to apply that rule. Deterministic, no API call.
    import json as _json

    def _safe_int(x, default=0):
        try:
            return int(x)
        except (ValueError, TypeError):
            return default

    def _title_only(row):
        total_w = len(str(row[cfg["corr_col"]]).split())
        title_w  = _safe_int(row.get("TitleWordCount"))
        ending_w = _safe_int(row.get("EndingWordCount"))
        other_w = 0
        other_json = str(row.get("OtherSpansJSON", "") or "")
        if other_json and other_json not in ("", "[]", "NA"):
            try:
                spans = _json.loads(other_json)
                for sp in spans:
                    if isinstance(sp, dict):
                        other_w += len(str(sp.get("text", "")).split())
                    elif isinstance(sp, str):
                        other_w += len(sp.split())
            except Exception:
                pass
        body_w = total_w - title_w - ending_w - other_w
        return title_w > 0 and body_w <= 3

    title_only_ids = [df.iloc[i][cfg["id_col"]] for i in range(len(df))
                      if _title_only(df.iloc[i])]
    if title_only_ids:
        print(f"  [trigger] Title or heading only detected for "
              f"{len(title_only_ids)} script(s): {title_only_ids}")
        mask_sent = sent["Identifier"].isin(title_only_ids)
        sent.loc[mask_sent, "ScriptClass"] = "Title or heading only"
        sent.loc[mask_sent, "InputQuality"] = "Title or heading only"
    # surface ScriptClass on texts so downstream scoring has one source
    # of truth alongside Concern1
    sc_map = sent.groupby("Identifier")["ScriptClass"].first().to_dict()
    df["ScriptClass"] = df[cfg["id_col"]].astype(str).map(
        {str(k): v for k, v in sc_map.items()}).fillna("TBD(engine)")

    # --- Stage 9: text-level classification (TB2, toggle-controlled) ---
    if "cp6_authorial" in _done:
        print(f"  [CP] TB2 authorial skipped — df loaded from checkpoint")
    elif cfg.get("stage_authorial", False):
        try:
            from text_level_classifier import add_text_level_columns, ALL_TB2_COLS

            # Initialise all TB2 output columns to None (idempotent if already present)
            for _col in ALL_TB2_COLS:
                if _col not in df.columns:
                    df[_col] = pd.array([None] * len(df), dtype=object)
                elif df[_col].dtype != object:
                    df[_col] = df[_col].astype(object)

            # Determine which scripts still need TB2 (None means not yet classified)
            _needs_tb2 = df["OnGenre"].isna()
            _n_todo = int(_needs_tb2.sum())
            _n_done_already = len(df) - _n_todo

            if _n_todo == 0:
                print(f"  [TB2] all {len(df)} scripts already classified (loaded from partial checkpoint)")
            else:
                if _n_done_already:
                    print(f"  [TB2] resuming — {_n_done_already} already done, {_n_todo} remaining")
                else:
                    print(f"  [TB2] classifying {_n_todo} scripts")

                _TB2_CHUNK = 50
                _todo_idx = df.index[_needs_tb2].tolist()
                for _ci_start in range(0, len(_todo_idx), _TB2_CHUNK):
                    _chunk_idx = _todo_idx[_ci_start:_ci_start + _TB2_CHUNK]
                    _chunk_df = df.loc[_chunk_idx].copy()
                    _chunk_result = add_text_level_columns(
                        _chunk_df,
                        prompt_text=cfg.get("prompt_text", ""),
                        cfg=cfg, api_key=cfg["api_key"]
                    )
                    for _col in ALL_TB2_COLS:
                        if _col in _chunk_result.columns:
                            df.loc[_chunk_idx, _col] = _chunk_result[_col]
                    _total_done = _n_done_already + _ci_start + len(_chunk_idx)
                    _ptag = f"cp6t_{_total_done:03d}"
                    _cp_save(_ptag, cfg, df=df)

            n_classified = df["OnGenre"].notna().sum() if "OnGenre" in df.columns else 0
            print(f"Text-level classification done ({n_classified} scripts).")
            _cp_save("cp6_authorial", cfg, df=df)
        except Exception as e:
            print(f"  [TB2] text-level classification failed: {e}; continuing")
    else:
        # Add null columns so schema is consistent regardless of toggle.
        # Note: OnPrompt/OnPromptReason/Recount/RecountReason removed in TB2.
        try:
            from text_level_classifier import ALL_TB2_COLS as _tb2_cols
        except Exception:
            _tb2_cols = [
                "OnGenre", "OnGenreReason", "TitleOnly", "CopyPrompt",
                "Scribed", "ScribedSignals", "SoC", "SoCReason",
                "Orientation", "Complication", "Resolution",
                "TextStructure", "TextStructureReason",
                "Ideas", "IdeasReason", "Theme", "ThemeReason",
                "Characters", "CharacterAnalysis", "CharacterAnalysisReason",
                "Setting", "SettingAnalysis", "SettingAnalysisReason",
            ]
        for _col in _tb2_cols:
            if _col not in df.columns:
                df[_col] = None

    # --- Stage 10: Pa scorer (TC5, quarantine-by-default) ---
    try:
        from pa_scorer import add_pa_columns
        df = add_pa_columns(df, html_col="Raw HTML", pa_col="Pa")
        n_html = (df.get("Raw HTML", pd.Series([])) != "").sum()
        print(f"Pa scorer done (HTML available for {n_html} scripts; "
              f"remainder = marker_assigned).")
    except Exception as e:
        print(f"  [TC5] Pa scorer failed: {e}; continuing")

    # --- TC2a: criteria-sum assertion check ---
    # Warn whenever a script's Total does not equal the sum of its criteria scores.
    CRITERIA_COLS = [
        "AU", "TS", "ID", "CS/PD", "Voc", "Coh", "Pa", "SS", "Pun", "Spell"
    ]
    _tc2a_cols = [c for c in CRITERIA_COLS if c in df.columns]
    if _tc2a_cols and "Total" in df.columns:
        _tc2a_warns = 0
        for _idx, _row in df.iterrows():
            try:
                crit_sum = sum(
                    float(_row[c]) for c in _tc2a_cols
                    if str(_row[c]) not in ("", "NA", "nan")
                )
                total = float(_row.get("Total", float("nan")))
                if abs(crit_sum - total) > 0.01:
                    print(f"  [TC2a warn] {_row.get(cfg['id_col'])}: "
                          f"Total {total} != criteria sum {crit_sum:.1f}")
                    _tc2a_warns += 1
            except (ValueError, TypeError):
                pass
        if _tc2a_warns == 0:
            print(f"TC2a: all {len(df)} scripts pass criteria-sum check.")
        else:
            print(f"TC2a: {_tc2a_warns} scripts have Total != criteria sum.")

    # --- Stage SF2: per-script SpellingFinal counts on texts ---
    try:
        import spelling_final as sf
        df = sf.add_spelling_counts(df, wm, id_col=cfg["id_col"])
        print(f"  [SF2] 11 Spell_* count columns added to texts.")
    except Exception as e:
        print(f"  [SF2] failed: {e}; continuing without spelling counts")

    # --- Stage BK2-counts: per-script break counts on texts ---
    try:
        import break_classifications as bc
        df = bc.add_break_counts(df, sent, id_col=cfg["id_col"])
        print(f"  [BK2] 7 break-count columns added to texts.")
    except Exception as e:
        print(f"  [BK2] count failed: {e}; continuing without break counts")

    # --- Stage SS-counts: per-script Sentence Structure counts on texts ---
    # Counts assessable sentences, correct/incorrect split, per-type pairs,
    # and structural variety. Inline time-slips forced to Fragment per
    # BK2-INV1. Mostly zero on local_only runs (Sentence type = TBD(engine))
    # but assessable counts are populated immediately.
    try:
        import ss_counts as ssc
        df = ssc.add_ss_counts(df, sent, id_col=cfg["id_col"])
        n_assess = int(df["SS_AssessableCount"].sum()) if "SS_AssessableCount" in df.columns else 0
        n_excl = int(df["SS_ExcludedCount"].sum()) if "SS_ExcludedCount" in df.columns else 0
        n_incorr = int(df["SS_IncorrectCount"].sum()) if "SS_IncorrectCount" in df.columns else 0
        print(f"  [SS] {n_assess} assessable, {n_excl} excluded, "
              f"{n_incorr} SS-Incorrect across {len(df)} scripts.")
    except Exception as e:
        print(f"  [SS] count failed: {e}; continuing without SS counts")

    # --- Punctuation counts (Phase PF) ---
    # Per-script Sentence / Other / Noun-cap blocks plus stray capitals, from
    # the sentence-level Punc boundary / Punc internal verdicts and the
    # word-level Cap class. Deterministic; safe to re-run.
    try:
        import punctuation_counts as pc
        df = pc.add_punctuation_counts(df, sent, wm, id_col=cfg["id_col"])
        sc = int(df["Pun_Sentence_Correct"].sum()) if "Pun_Sentence_Correct" in df.columns else 0
        si = int(df["Pun_Sentence_Incorrect"].sum()) if "Pun_Sentence_Incorrect" in df.columns else 0
        nc = int(df["Pun_NounCaps_Incorrect"].sum()) if "Pun_NounCaps_Incorrect" in df.columns else 0
        print(f"  [PF] sentence punct {sc} correct / {si} incorrect, "
              f"{nc} noun-cap errors across {len(df)} scripts.")
    except Exception as e:
        print(f"  [PF] count failed: {e}; continuing without punctuation counts")

    # --- Vocabulary counts + band (Phase VC) ---
    # Aggregates word-level LexicalPrecision into per-script counts and applies
    # the Vocabulary ladder (0-4 deterministic; band 5 is the LLM
    # appropriateness/effectiveness gate, not yet built). Deterministic, free.
    try:
        import vocab_counts as vc
        df = vc.add_vocab_columns(df, wm, id_col=cfg["id_col"])
        bd = df["Voc_Band"].value_counts().sort_index().to_dict() if "Voc_Band" in df.columns else {}
        cand = int(df["Voc_Band5Candidate"].sum()) if "Voc_Band5Candidate" in df.columns else 0
        print(f"  [VC] vocabulary bands {bd}; {cand} band-5 candidates (pending appropriateness check).")
    except Exception as e:
        print(f"  [VC] count failed: {e}; continuing without vocabulary counts")

    # --- Cohesion counts + band (Phase CC) ---
    # Aggregates sentence-level tense/pronoun/grammar errors and word-level
    # CohesionRole into per-script counts and applies the Cohesion ladder
    # (0-3 deterministic; band 4 is the LLM 'flows well' gate, not yet built).
    # Two-tier error model: definitive errors gate 2->3; light errors gate 3->4.
    # Deterministic, free.
    try:
        import coh_counts as cc
        df = cc.add_coh_columns(df, sent, wm, id_col=cfg["id_col"])
        bd = df["Coh_Band"].value_counts().sort_index().to_dict() if "Coh_Band" in df.columns else {}
        cand = int(df["Coh_Band4Candidate"].sum()) if "Coh_Band4Candidate" in df.columns else 0
        print(f"  [CC] cohesion bands {bd}; {cand} band-4 candidates (pending flows-well check).")
    except Exception as e:
        print(f"  [CC] count failed: {e}; continuing without cohesion counts")

    # --- Save ---
    o = cfg["out_dir"]
    # A3: drop boundary-retained text from output (still built internally above)
    # TC3: drop yrlev pass-through column from output
    # Pa1 dedup: HTML is copied to Raw HTML at load time; drop the original to avoid
    #   duplicate content columns in texts.csv
    # Identifier dedup: Identifier is always == Research ID on the texts table; drop it
    _drop_from_texts = [
        "Corrected text (boundaries retained)",
        "yrlev",
        "HTML",        # duplicate of Raw HTML (Pa1 passthrough)
        "Identifier",  # duplicate of Research ID on texts table
    ]
    wm.drop(columns=["_ap"], errors="ignore").to_csv(f"{o}/word_map.csv", index=False)
    sent.to_csv(f"{o}/sentences.csv", index=False)
    df.drop(columns=_drop_from_texts, errors="ignore").to_csv(f"{o}/texts.csv", index=False)
    print(f"\nSaved word_map.csv, sentences.csv, texts.csv to {o}")
    print("Reminder: Concern1 is advisory and must be reviewed by a human; "
          "it never feeds any score.")

    # --- Cost summary ---
    try:
        from cost_tracker import tracker
        tracker.print_summary()
        print(f"Estimated total cost: ${tracker.total_cost():.4f} USD")
    except Exception as e:
        print(f"  [cost] summary unavailable: {e}")


if __name__ == "__main__":
    import sys
    _presets = list(_PRESETS)
    _arg = sys.argv[1] if len(sys.argv) > 1 else None
    if _arg and _arg not in _presets:
        print(f"Unknown preset {_arg!r}. Available: {_presets}")
        sys.exit(1)
    main(preset=_arg)
