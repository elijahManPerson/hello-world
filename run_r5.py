"""
run_r5.py
=========
Run the AES pipeline (r5, Batch 1 fixes applied) on the 21-script CSV
using the real Claude API corrector.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 run_r5.py

Outputs (written to the same directory as this script):
    output_word_map_claude_r5.csv
    output_sent_df_claude_r5.csv
    output_texts_claude_r5.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import aes_canonical as aes
from corrector_claude import make_claude_corrector

DATA_PATH = "/root/.claude/uploads/a7b9fba6-c46a-4166-abd1-4f5b41a9969b/f1312d94-Data_for_Testing_avg_short.csv"
OUT_DIR   = os.path.dirname(__file__)
TAG       = "r5"

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} scripts.")

    corrector = make_claude_corrector(
        model="claude-opus-4-7",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_retries=3,
        retry_base_delay=2.0,
    )

    print("Starting pipeline — this will take a few minutes...")
    result = aes.run_pipeline(
        df,
        corrector=corrector,
        raw_col="Raw text",
        id_col="Research ID",
    )

    df_map   = result["df_map"]
    df_sent  = result["sent_df"]
    df_texts = result["df_texts"]

    df_map.to_csv(  f"{OUT_DIR}/output_word_map_claude_{TAG}.csv", index=False)
    df_sent.to_csv( f"{OUT_DIR}/output_sent_df_claude_{TAG}.csv",  index=False)
    df_texts.to_csv(f"{OUT_DIR}/output_texts_claude_{TAG}.csv",    index=False)

    print(f"word_map : {len(df_map)} rows")
    print(f"sent_df  : {len(df_sent)} rows")
    print(f"texts    : {len(df_texts)} rows")
    print("Done.")

if __name__ == "__main__":
    main()
