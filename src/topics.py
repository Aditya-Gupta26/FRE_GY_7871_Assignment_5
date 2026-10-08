"""Topic models for Q1.

Main: BERTopic (Grootendorst 2022, arXiv:2203.05794): sentence embeddings
(all-MiniLM-L6-v2, 384-d) -> UMAP (5 dims, 15 neighbours, cosine) -> HDBSCAN clusters ->
class-based TF-IDF for the topic words, W_{t,c} = tf_{t,c} * log(1 + A / tf_t) (paper eq. 2;
A = average number of words per class). Chosen because the texts are short (tweets, comments,
headlines), where bag-of-words LDA has little co-occurrence to work with.
Baseline: LDA (Blei, Ng and Jordan 2003), the model of the class reading Bybee, Kelly, Manela
and Xiu (2024, JF), who measure "news attention" as each topic's share of text over time.
Coherence: C_V and C_NPMI (Roder, Both and Hinneburg 2015) from gensim's CoherenceModel on the
same tokenised texts; C_NPMI is reported too because Roder has said gensim's C_V does not
reproduce the paper (Palmetto issue #76).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

STOP_EXTRA = {"http", "https", "user", "amp", "rt", "just", "like", "people", "think", "don", "dont",
              "im", "it", "thats", "doesn", "didn", "isn", "ve", "ll", "re", "going", "know", "really",
              "got", "get", "say", "said", "want", "make", "way", "thing", "things", "lol", "yeah",
              "did", "does", "let", "need", "good", "right", "time", "new", "news", "says"}
TOKEN = re.compile(r"[a-z][a-z'\-]{2,}")


def tokenize(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [w for w in TOKEN.findall((text or "").lower().replace("'", ""))
            if w not in ENGLISH_STOP_WORDS and w not in STOP_EXTRA]


def stratified_sample(meta: pd.DataFrame, n: int, by: list[str], seed: int) -> np.ndarray:
    """Row positions sampled proportionally within each `by` cell (keeps every day/source)."""
    rng = np.random.default_rng(seed)
    frac = min(1.0, n / len(meta))
    pos = []
    for _, g in meta.groupby(by, observed=True):
        k = max(1, int(round(len(g) * frac)))
        pos.extend(rng.choice(g.index.to_numpy(), size=min(k, len(g)), replace=False))
    return np.sort(np.array(pos))


def fit_bertopic(docs: list[str], emb: np.ndarray, seed: int, min_cluster: int, min_samples: int = 15):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = sorted(set(ENGLISH_STOP_WORDS) | STOP_EXTRA)
    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=10,
                                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"),
        top_n_words=10, calculate_probabilities=False, verbose=True)
    topics, _ = model.fit_transform(docs, embeddings=emb)
    return model, np.array(topics)


def topic_words(model, k: int = 10) -> dict[int, list[str]]:
    return {t: [w for w, _ in model.get_topic(t)[:k]] for t in model.get_topics() if t != -1}


def coherence(words: dict, texts_tok: list[list[str]], kinds=("c_npmi", "c_v")) -> dict:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    dic = Dictionary(texts_tok)
    tops = [[w for w in ws if w in dic.token2id] for ws in words.values()]
    tops = [t for t in tops if len(t) >= 3]
    out = {}
    for kd in kinds:
        cm = CoherenceModel(topics=tops, texts=texts_tok, dictionary=dic, coherence=kd, processes=1)
        out[kd] = float(cm.get_coherence())
    return out


def lda_sweep(texts_tok: list[list[str]], ks=(10, 15, 20, 25, 30), seed: int = 7871) -> pd.DataFrame:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    dic = Dictionary(texts_tok)
    dic.filter_extremes(no_below=10, no_above=0.5)
    bow = [dic.doc2bow(t) for t in texts_tok]
    rows, models = [], {}
    for k in ks:
        lda = LdaModel(bow, id2word=dic, num_topics=k, passes=5, random_state=seed,
                           chunksize=4000)
        words = {i: [w for w, _ in lda.show_topic(i, 10)] for i in range(k)}
        rows.append({"K": k, **coherence(words, texts_tok)})
        models[k] = (lda, words)
        print("LDA K", k, rows[-1], flush=True)
    return pd.DataFrame(rows), models, dic


def match_topics(emb: np.ndarray, lab_a: np.ndarray, lab_b: np.ndarray) -> dict[int, int]:
    """Map each topic of run B to the run-A topic with the most similar centroid (cosine)."""
    def cents(lab):
        ids = [t for t in np.unique(lab) if t != -1]
        C = np.vstack([emb[lab == t].mean(0) for t in ids])
        return ids, C / np.linalg.norm(C, axis=1, keepdims=True)
    ia, A = cents(lab_a)
    ib, B = cents(lab_b)
    S = B @ A.T
    return {b: ia[int(S[i].argmax())] for i, b in enumerate(ib)}
