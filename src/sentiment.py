"""Tone and stance scorers (adapted from Assignment 4).

- VADER (Hutto and Gilbert 2014, ICWSM): rule-based, built for social media; the compound
  score is normalised to [-1, 1] (score / sqrt(score^2 + 15)), cut-offs +-0.05.
- Twitter-RoBERTa, cardiffnlp/twitter-roberta-base-sentiment-latest (TimeLMs, Loureiro et al.
  2022; fine-tuned on TweetEval, Barbieri et al. 2020): ~124M tweets, labels negative /
  neutral / positive.
- Zero-shot NLI stance, MoritzLaurer/deberta-v3-base-zeroshot-v2.0 (Laurer et al. 2023),
  run with multi_label=True as the model author advises (each label scored on its own,
  not softmaxed against the others).
The class readings on tool choice (Carvalho and Plastino 2020; Psomakelis et al. 2014; the
autism-awareness tool-comparison slides) show tools disagree, so every tool is validated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ROBERTA = "cardiffnlp/twitter-roberta-base-sentiment-latest"
NLI = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


def vader_scores(texts: list[str]) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    out = [sia.polarity_scores(t or "") for t in texts]
    return pd.DataFrame(out).rename(columns={"compound": "vader_compound", "pos": "vader_pos",
                                             "neg": "vader_neg", "neu": "vader_neu"})


class Transformer3:
    """Batched 3-class classifier; label order read from the model config."""

    def __init__(self, name: str = ROBERTA, max_len: int = 128, batch: int = 64):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
        self.max_len, self.batch = max_len, batch
        id2 = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.order = [id2[i] for i in range(len(id2))]

    @torch.no_grad()
    def __call__(self, texts: list[str], desc: str = "") -> pd.DataFrame:
        probs = []
        idx = np.argsort([len(t or "") for t in texts])
        for i in tqdm(range(0, len(texts), self.batch), desc=desc, mininterval=30):
            chunk = [texts[j] or "" for j in idx[i:i + self.batch]]
            enc = self.tok(chunk, truncation=True, max_length=self.max_len, padding=True,
                           return_tensors="pt").to(DEVICE)
            probs.append(torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy())
        P = np.vstack(probs) if probs else np.zeros((0, 3))
        out = np.empty_like(P)
        out[idx] = P
        return pd.DataFrame(out, columns=self.order)


def roberta_scores(texts: list[str]) -> pd.DataFrame:
    P = Transformer3(ROBERTA)(texts, desc="roberta")
    return P.rename(columns={"negative": "rob_neg", "neutral": "rob_neu", "positive": "rob_pos"})


class NLIStance:
    """Zero-shot stance with an NLI model: P(entailment) of each hypothesis on its own
    (multi_label=True), entailment vs not-entailment from the model's two-way head."""

    HYP = {"D": "This text supports the Democrats or criticizes Republicans or Trump.",
           "R": "This text supports Republicans or Trump or criticizes the Democrats."}

    def __init__(self, name: str = NLI, max_len: int = 256, batch: int = 32):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
        id2 = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ent = [i for i, v in id2.items() if v.startswith("entail")][0]
        self.max_len, self.batch = max_len, batch

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> pd.DataFrame:
        out = {}
        order = np.argsort([len(t or "") for t in texts])     # length-sorted batches, less padding
        for k, h in self.HYP.items():
            ps = []
            for i in tqdm(range(0, len(texts), self.batch), desc=f"nli-{k}", mininterval=30):
                chunk = [texts[j] or "" for j in order[i:i + self.batch]]
                enc = self.tok(chunk, [h] * len(chunk), truncation="only_first",
                               max_length=self.max_len, padding=True, return_tensors="pt").to(DEVICE)
                ps.append(torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.ent].cpu().numpy())
            p_sorted = np.concatenate(ps) if ps else np.zeros(0)
            p_all = np.empty_like(p_sorted)
            p_all[order] = p_sorted
            out[f"nli_{k}"] = p_all
        df = pd.DataFrame(out)
        df["nli_s"] = df["nli_D"] - df["nli_R"]
        return df
