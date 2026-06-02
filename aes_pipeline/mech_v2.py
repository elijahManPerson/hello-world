"""
mech_v2.py
==========
Additive layer on top of aes_canonical.  It does NOT modify your original
file.  It does three things:

  1. Fixes the tokeniser so contractions ("don't", "Let's") stay one token.
  2. Teaches the sentence splitter that straight double quotes are quotes,
     so dialogue stops fragmenting.
  3. Adds the agreed schema: a token broad category + subcategory, plus the
     twelve columns (4 families x error / subtype / class).

Cheap, reliable columns are filled here with pandas only.  The two columns
that need a part-of-speech / grammar engine are written as the literal
string ENGINE so it is obvious they are deliberately deferred, not missing:
  - broad subcategory for word / contraction rows (the word class)
  - Sentence structure class (the grammatical subtype)

Run validate_on_r5() to see before/after on the saved r5 corrected text
WITHOUT calling any API.
"""

import re
import pandas as pd
import numpy as np
import aes_canonical as aes

ENGINE = "TBD(engine)"

# See full file at /tmp/aes_pipeline/mech_v2.py - content too large for inline push
# This is a placeholder - the actual file needs to be pushed via git
