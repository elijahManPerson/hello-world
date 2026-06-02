"""
contractions_lookup.py
======================
Deterministic POS overrides for contractions and common misformed tokens.
Used by mech_v2.apply_contraction_pos_override().
"""

CONTRACTION_POS = {
    # --- First-person pronoun + auxiliary/copula ---
    "i'm":    ("pronoun", "subject"),
    "im":     ("pronoun", "subject"),
    "i'll":   ("pronoun", "subject"),
    "i've":   ("pronoun", "subject"),
    "i'd":    ("pronoun", "subject"),

    # --- Other pronoun + auxiliary/copula ---
    "he's":   ("pronoun", "subject"),
    "he'd":   ("pronoun", "subject"),
    "he'll":  ("pronoun", "subject"),
    "she's":  ("pronoun", "subject"),
    "she'd":  ("pronoun", "subject"),
    "she'll": ("pronoun", "subject"),
    "it's":   ("pronoun", "subject"),
    "it'll":  ("pronoun", "subject"),
    "we're":  ("pronoun", "subject"),
    "we've":  ("pronoun", "subject"),
    "we'll":  ("pronoun", "subject"),
    "we'd":   ("pronoun", "subject"),
    "they're":("pronoun", "subject"),
    "they've":("pronoun", "subject"),
    "they'll":("pronoun", "subject"),
    "they'd": ("pronoun", "subject"),
    "you're": ("pronoun", "subject"),
    "you've": ("pronoun", "subject"),
    "you'll": ("pronoun", "subject"),
    "you'd":  ("pronoun", "subject"),
    "that's": ("pronoun", "subject"),
    "there's":("pronoun", "subject"),
    "let's":  ("pronoun", "subject"),
    "who's":  ("pronoun", "subject"),
    "what's": ("pronoun", "subject"),
    "where's":("pronoun", "subject"),
    "here's": ("pronoun", "subject"),

    # --- n't negation: auxiliary verb ---
    "didn't":    ("auxiliary verb", "negation"),
    "couldn't":  ("auxiliary verb", "negation"),
    "wouldn't":  ("auxiliary verb", "negation"),
    "shouldn't": ("auxiliary verb", "negation"),
    "won't":     ("auxiliary verb", "negation"),
    "don't":     ("auxiliary verb", "negation"),
    "doesn't":   ("auxiliary verb", "negation"),
    "wasn't":    ("auxiliary verb", "negation"),
    "weren't":   ("auxiliary verb", "negation"),
    "isn't":     ("auxiliary verb", "negation"),
    "aren't":    ("auxiliary verb", "negation"),
    "hasn't":    ("auxiliary verb", "negation"),
    "haven't":   ("auxiliary verb", "negation"),
    "hadn't":    ("auxiliary verb", "negation"),
    "can't":     ("auxiliary verb", "negation"),
    "cannot":    ("auxiliary verb", "negation"),
    "mustn't":   ("auxiliary verb", "negation"),
    "needn't":   ("auxiliary verb", "negation"),
    "daren't":   ("auxiliary verb", "negation"),
    "mightn't":  ("auxiliary verb", "negation"),
    "shan't":    ("auxiliary verb", "negation"),

    # --- Student misspelling variants (no apostrophe) ---
    "dont":     ("auxiliary verb", "negation"),
    "didnt":    ("auxiliary verb", "negation"),
    "couldnt":  ("auxiliary verb", "negation"),
    "wouldnt":  ("auxiliary verb", "negation"),
    "wasnt":    ("auxiliary verb", "negation"),
    "isnt":     ("auxiliary verb", "negation"),
    "havent":   ("auxiliary verb", "negation"),
    "hasnt":    ("auxiliary verb", "negation"),
    "couldent": ("auxiliary verb", "negation"),
    "werent":   ("auxiliary verb", "negation"),
    "wont":     ("auxiliary verb", "negation"),
    "cant":     ("auxiliary verb", "negation"),
    "arent":    ("auxiliary verb", "negation"),
    "hadnt":    ("auxiliary verb", "negation"),
}
