"""Optional keras backend: use the trained per-criterion models as the scorer.

This bridges the September-2024 notebook models (``model_<criterion>.keras``)
into the engine: it reconstructs the *exact* feature pipeline the notebook
trained on, predicts each criterion, and hands the scores to the fit layer via
:func:`criteria_engine.predict_criteria`.

IMPORTANT - the saved vectorizer requirement
---------------------------------------------
The notebook input is ``hstack([scaled[WordCount, TextLength,
AvgSentenceLength], tfidf(Raw text)])`` = 1003 features. The TF-IDF vocabulary
is therefore part of the model. The original training run saved
``saved_scaler.pkl`` but NOT the TF-IDF vectorizer, so the original ``.keras``
weights cannot be reproduced exactly. To use this backend you must re-run
training with the vectorizer-saving step (see ``notebook_patch`` below), which
produces a matched ``tfidf_vectorizer.pkl`` + ``saved_scaler.pkl`` +
``model_*.keras`` set.

Heavy deps (tensorflow/keras, scikit-learn, joblib) are imported lazily so the
core engine stays dependency-free.
"""

from __future__ import annotations

import os

from .ranges import CRITERIA, clamp
from .engine import predict_criteria

# Criterion key -> saved model filename stem (matches the notebook's
# sanitize_filename over its model dict keys).
_MODEL_STEM: dict[str, str] = {
    "AU": "model_AU",
    "TS": "model_TS",
    "ID": "model_ID",
    "CS/PD": "model_CS_PD",
    "Voc": "model_Vocabulary",
    "Coh": "model_Coh",
    "Pa": "model_Pa",
    "SS": "model_SentenceStructure",
    "Pun": "model_Punctuation",
    "Spell": "model_Spelling",
}


def _notebook_features(text: str, word_count: int | None):
    """Replicate the notebook's three engineered features for one script."""
    text = text or ""
    text_length = len(text)
    wc = word_count if word_count is not None else len(text.split())
    avg_sentence_length = len(text.split()) / (text.count(".") + 1) if text.split() else 0.0
    return wc, text_length, avg_sentence_length


class KerasCriterionScorer:
    """Loads the trained vectorizer, scaler and per-criterion keras models."""

    def __init__(
        self,
        model_dir: str,
        tfidf_path: str,
        scaler_path: str,
    ):
        import joblib  # lazy
        from tensorflow import keras  # lazy

        self.tfidf = joblib.load(tfidf_path)
        self.scaler = joblib.load(scaler_path)
        self.models = {}
        missing = []
        for crit, stem in _MODEL_STEM.items():
            path = os.path.join(model_dir, f"{stem}.keras")
            if os.path.exists(path):
                self.models[crit] = keras.models.load_model(path)
            else:
                missing.append(path)
        if missing:
            raise FileNotFoundError(
                "missing model files:\n  " + "\n  ".join(missing)
            )

    def score(self, text: str, word_count: int | None = None) -> dict[str, int]:
        """Predict the ten criterion scores for one script (clamped to range)."""
        import numpy as np  # lazy

        wc, text_length, avg_sl = _notebook_features(text, word_count)
        tfidf_vec = self.tfidf.transform([text or ""]).toarray()
        scaled = self.scaler.transform([[wc, text_length, avg_sl]])
        x = np.hstack([scaled, tfidf_vec])

        out: dict[str, int] = {}
        for crit, model in self.models.items():
            pred = float(np.ravel(model.predict(x, verbose=0))[0])
            out[crit] = clamp(crit, pred)
        return out

    def predict(self, text: str, word_count: int | None = None, **kwargs) -> dict:
        """Score with the keras models, then run the fit layer."""
        scores = self.score(text, word_count=word_count)
        return predict_criteria(
            text=text, word_count=word_count, criterion_scores=scores, **kwargs
        )


# Notebook cell (paste after the TF-IDF/scaler fit cell) to make future runs
# reusable by saving the fitted vectorizer alongside the scaler:
notebook_patch = '''
# Persist the fitted vectorizer + scaler so the saved .keras models are reusable
import joblib
joblib.dump(tfidf,  '/content/drive/MyDrive/tfidf_vectorizer.pkl')
joblib.dump(scaler, '/content/drive/MyDrive/saved_scaler.pkl')
print('saved tfidf_vectorizer.pkl and saved_scaler.pkl')
'''
