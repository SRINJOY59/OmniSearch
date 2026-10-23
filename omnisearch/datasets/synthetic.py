"""Synthetic data with controlled properties, for systems benchmarks.

Real embeddings are not needed to measure index and optimizer behaviour, but
the data must have structure: vectors are drawn around cluster centres, text
is drawn from per-cluster topic vocabularies (so keyword and vector scores are
correlated, as with real multimodal data), and attributes have exactly known
selectivities.
"""
from __future__ import annotations

import numpy as np

from omnisearch.util import normalize_rows


def clustered_vectors(n: int, dim: int, n_clusters: int = 64, spread: float = 0.35, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = normalize_rows(rng.normal(size=(n_clusters, dim)))
    labels = rng.integers(0, n_clusters, n)
    x = centers[labels] + spread * rng.normal(size=(n, dim)) / np.sqrt(dim) * 4
    return normalize_rows(x).astype(np.float32), labels, centers


def queries_like(centers, nq: int, spread: float = 0.35, seed: int = 1):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(centers), nq)
    dim = centers.shape[1]
    q = centers[labels] + spread * rng.normal(size=(nq, dim)) / np.sqrt(dim) * 4
    return normalize_rows(q).astype(np.float32), labels


def topic_corpus(labels, vocab_size: int = 20_000, topic_words: int = 60, doc_len=(8, 40), seed: int = 2):
    """Zipfian background words + words from the document's cluster topic."""
    rng = np.random.default_rng(seed)
    n_clusters = int(labels.max()) + 1
    vocab = np.array([f"w{i}" for i in range(vocab_size)])
    zipf = 1.0 / np.arange(1, vocab_size + 1) ** 1.05
    zipf /= zipf.sum()
    topics = [rng.choice(vocab_size, topic_words, replace=False) for _ in range(n_clusters)]
    texts = []
    for lab in labels:
        length = int(rng.integers(*doc_len))
        n_topic = max(1, int(length * 0.35))
        words = np.concatenate([rng.choice(vocab_size, length - n_topic, p=zipf),
                                rng.choice(topics[lab], n_topic)])
        texts.append(" ".join(vocab[words]))
    return texts, [vocab[t] for t in topics]


def topic_queries(query_labels, topics, n_words=(2, 4), seed: int = 3):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(topics[lab], int(rng.integers(n_words[0], n_words[1] + 1)), replace=False))
            for lab in query_labels]


def attributes(labels, seed: int = 4) -> dict:
    """bucket: uniform 0..999 (``bucket < s*1000`` has selectivity s, independent of
    the vectors). group: equal to the cluster for half the docs - filters on it are
    correlated with the query's neighbourhood. price: log-normal, skewed."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    group = np.where(rng.random(n) < 0.5, labels, rng.integers(0, labels.max() + 1, n))
    return {
        "bucket": rng.integers(0, 1000, n),
        "group": group,
        "price": np.round(rng.lognormal(3.5, 0.9, n), 2),
    }
