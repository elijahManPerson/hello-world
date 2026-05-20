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

Edit the CONFIG block, then run:  python3 run_all.py

Each engine stage has its own switch. Set a stage to "mock" first to see
what it would do at zero cost, then "run" it for real once you are happy.
Stages already proven on data (correction is yours; the cheap layers) just
run. The order matters and is enforced here so you do not have to think
about it.
"""

import os
import re
import pandas as pd

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
    lex = pd.read_csv(lexicon_path, keep_default_na=False)
    lex = lex[lex["DifficultyCategory"] != ""]        # exclude set-aside rows
    lex["_key"] = lex["Word"].str.lower().str.strip()
    lex = lex.drop_duplicates("_key", keep="first")
    cat_map  = lex.set_index("_key")["DifficultyCategory"]
    conf_map = lex.set_index("_key")["Confidence"]

    df = wm.copy()
    df["SpellDifficulty"] = "NA"
    df["SpellDifficultyConfidence"] = "NA"
    wordish = df["TokenCategory"].isin(["word", "contraction"])
    keys = (df.loc[wordish, "corr_token"]
            .fillna("").astype(str)
            .str.lower()
            .str.replace(r"[^a-z]", "", regex=True))
    df.loc[wordish, "SpellDifficulty"] = keys.map(cat_map).fillna("NA").values
    df.loc[wordish, "SpellDifficultyConfidence"] = keys.map(conf_map).fillna("NA").values
    return df


def main(cfg=CONFIG):
    import mech_v2 as m
    import engine_prototype as ep
    import concern1_detector as c1

    os.makedirs(cfg["out_dir"], exist_ok=True)

    # keep_default_na=False preserves the deliberate "NA" sentinel so it is
    # never silently converted to NaN on reload (BUILD_BRIEF issue #1).
    df = pd.read_csv(cfg["input_csv"], keep_default_na=False)
    print(f"Loaded {len(df)} scripts from {cfg['input_csv']}")

    # --- Stage 1: correction (produces the corrected-text column) ---
    if cfg["corr_col"] not in df.columns and cfg["stage_correct"] != "skip":
        from corrector_claude import make_claude_corrector
        corrector = make_claude_corrector(model=cfg["model_correct"],
                                          api_key=cfg["api_key"])
        if cfg["stage_correct"] == "mock":
            print("MOCK correct: would correct",
                  len(df), "scripts with", cfg["model_correct"])
            df[cfg["corr_col"]] = df[cfg["raw_col"]]      # placeholder
        else:
            df[cfg["corr_col"]] = df[cfg["raw_col"]].astype(str).map(corrector)
            print("Correction done.")

    # --- Stage 2: title / ending / non-story detection ---
    if cfg["stage_artifacts"] in ("run", "mock"):
        import title_detector as td
        df = td.run_artifacts(df, id_col=cfg["id_col"],
                              raw_col=cfg["raw_col"],
                              model=cfg["model_artifacts"],
                              api_key=cfg["api_key"],
                              mock=(cfg["stage_artifacts"] == "mock"))
        print("Artifact stage done"
              + (" (MOCK)" if cfg["stage_artifacts"] == "mock" else ""))

    # --- Stage 3: fixed word layer + 12 cheap columns (always) ---
    wm = m.run_layer(df, id_col=cfg["id_col"], raw_col=cfg["raw_col"],
                     corr_col=cfg["corr_col"])
    print(f"Word layer: {len(wm)} token rows.")

    # --- Section 8: boundary-retained text (local, free, no API) ---
    br_texts = m.build_boundary_retained_texts(
        wm, texts=df, corr_col=cfg["corr_col"], id_col=cfg["id_col"])
    df["Corrected text (boundaries retained)"] = (
        df[cfg["id_col"]].astype(str).map(br_texts))
    print(f"Boundary-retained text built for {len(br_texts)} scripts.")

    # --- Stage 4: word class (engine selectable) ---
    _wc_engine = cfg.get("wordclass_engine", "spacy")
    if cfg["stage_wordclass"] == "run":
        if _wc_engine == "claude":
            wm = ep.fill_wordclass_with_claude(
                wm, texts=df, model=cfg["model_wordclass"],
                api_key=cfg["api_key"],
                corr_col=cfg["corr_col"], id_col=cfg["id_col"])
            print(f"Word class filled (Claude, {cfg['model_wordclass']}).")
        else:
            wm = ep.fill_with_spacy(wm)
            print("Word class filled (spaCy, local).")
    elif cfg["stage_wordclass"] == "mock":
        if _wc_engine == "claude":
            ep.fill_wordclass_with_claude(wm, texts=df, mock=True)
        else:
            ep.fill_with_spacy(wm, mock=True)

    # --- Stage 5: grammar subtype (Claude, only the edits) ---
    if cfg["stage_grammar"] == "run":
        wm = ep.fill_with_claude(wm, model=cfg["model_grammar"],
                                 api_key=cfg["api_key"])
        print("Grammar subtype filled.")
    elif cfg["stage_grammar"] == "mock":
        ep.fill_with_claude(wm, mock=True)

    # --- Stage 5b: word-difficulty lexicon join (local, free) ---
    if cfg.get("difficulty_lexicon"):
        wm = _join_difficulty(wm, cfg["difficulty_lexicon"])
        n_matched = (wm["SpellDifficulty"] != "NA").sum()
        print(f"Word difficulty joined ({n_matched} tokens matched).")

    # --- Stage 6: sentence layer (uses the engine-enriched word map) ---
    sent = m.run_sentence_layer(df, id_col=cfg["id_col"],
                                raw_col=cfg["raw_col"],
                                corr_col=cfg["corr_col"],
                                precomputed_wm=wm)
    print(f"Sentence layer: {len(sent)} sentence rows.")

    # --- Stage 7: sentence type + pronoun reference (Claude) ---
    if cfg["stage_sentence"] == "run":
        sent = ep.fill_sentence_engine(sent, model=cfg["model_sentence"],
                                       api_key=cfg["api_key"])
        print("Sentence type + pronoun reference filled.")
    elif cfg["stage_sentence"] == "mock":
        ep.fill_sentence_engine(sent, mock=True)

    # --- Stage 8: welfare detector (Claude, advisory, isolated) ---
    if cfg["stage_concern1"] in ("run", "mock"):
        df = c1.run_concern1(df, id_col=cfg["id_col"], raw_col=cfg["raw_col"],
                             model=cfg["model_concern1"],
                             api_key=cfg["api_key"],
                             mock=(cfg["stage_concern1"] == "mock"))
        print("Concern 1 stage done"
              + (" (MOCK)" if cfg["stage_concern1"] == "mock" else ""))

    # --- Save ---
    o = cfg["out_dir"]
    wm.to_csv(f"{o}/word_map.csv", index=False)
    sent.to_csv(f"{o}/sentences.csv", index=False)
    df.to_csv(f"{o}/texts.csv", index=False)
    print(f"\nSaved word_map.csv, sentences.csv, texts.csv to {o}")
    print("Reminder: Concern1 is advisory and must be reviewed by a human; "
          "it never feeds any score.")


if __name__ == "__main__":
    main()
