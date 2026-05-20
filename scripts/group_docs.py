"""
Group a flat list[str] document pkl into a list[list[str]] pkl where each group
has the form [context, doc].  For each source document, --num_docs context
documents are selected (randomly or by BM25 similarity) and concatenated into a
single context string stored as group[0].  When num_docs=1 the context is a
single document.

Usage:
    python group_docs.py \
        --source_pkl data/ms_100M/mdata_msmarco_100M.pkl \
        --dest_pkl   data/ms_100M/mdata_msmarco_grouped.pkl \
        --strategy   bm25 \
        --num_docs   1

Note: BM25 indexing over large corpora (>100K docs) takes several minutes.
"""

import argparse
import pickle
import random as _random


def load_pkl(path: str) -> list[str]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, list) and all(isinstance(d, str) for d in data), \
        f"{path} must be a list[str]"
    return data


def save_pkl(path: str, data: list[list[str]]) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def group_random(docs: list[str], num_docs: int) -> list[list[str]]:
    """
    For each doc, sample num_docs uniformly random other docs and concatenate
    them as the context string.
    Each group: [context, doc].
    """
    n = len(docs)
    result = []
    for i in range(n):
        pool = list(range(n))
        pool.remove(i)
        ctx_indices = _random.sample(pool, num_docs)
        context = "\n".join(docs[j] for j in ctx_indices)
        result.append([context, docs[i]])
    return result


def group_bm25(docs: list[str], num_docs: int) -> list[list[str]]:
    """
    For each doc, find its top num_docs most BM25-similar other docs and
    concatenate them as the context string.
    Each group: [context, doc].
    """
    from rank_bm25 import BM25Okapi
    import numpy as np

    print("Tokenizing corpus...")
    tokenized = [doc.lower().split() for doc in docs]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)

    n = len(docs)
    result = []
    for i in range(n):
        scores = bm25.get_scores(tokenized[i])
        scores[i] = -np.inf  # exclude self
        top_indices = np.argpartition(scores, -num_docs)[-num_docs:]
        # Sort by descending score so highest-similarity doc comes first
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        context = "\n".join(docs[j] for j in top_indices)
        result.append([context, docs[i]])

        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1}/{n}")
    return result


def group_bm25_clusters(docs: list[str], num_docs: int, n_clusters: int) -> list[list[str]]:
    """
    Cluster docs into n_clusters clusters via TF-IDF + k-means (word-overlap
    based).  For each cluster, identify the num_docs most central docs
    (highest mean cosine similarity to all other cluster members) and
    concatenate them as the shared context for every doc in that cluster,
    including the central docs themselves.
    Each group: [context, doc].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50_000)
    X = vectorizer.fit_transform(docs)

    print(f"Clustering into {n_clusters} clusters...")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=3)
    labels = km.fit_predict(X)

    print("Finding cluster centroids...")
    result = [None] * len(docs)
    for cluster_id in range(n_clusters):
        member_indices = np.where(labels == cluster_id)[0]
        if len(member_indices) == 0:
            continue

        X_cluster = X[member_indices]
        # Mean cosine similarity of each member to all other cluster members
        sim = cosine_similarity(X_cluster)
        np.fill_diagonal(sim, 0.0)
        mean_sim = sim.mean(axis=1)

        k = min(num_docs, len(member_indices))
        central_local = np.argpartition(mean_sim, -k)[-k:]
        central_local = central_local[np.argsort(mean_sim[central_local])[::-1]]
        central_indices = member_indices[central_local]
        context = "\n".join(docs[j] for j in central_indices)

        for idx in member_indices:
            result[idx] = [context, docs[idx]]

        if (cluster_id + 1) % max(1, n_clusters // 10) == 0:
            print(f"  {cluster_id + 1}/{n_clusters} clusters done")

    return result


def group_echo(docs: list[str], num_docs: int) -> list[list[str]]:
    """
    For each doc, use the doc itself as context.  num_docs copies of the doc
    are concatenated to form the context string.
    Each group: [context, doc].
    """
    return [
        ["\n".join([doc] * num_docs), doc]
        for doc in docs
    ]


def main():
    parser = argparse.ArgumentParser(description="Group flat doc pkl into contextualized pairs.")
    parser.add_argument("--source_pkl",  required=True, help="Input list[str] pkl file.")
    parser.add_argument("--dest_pkl",    required=True, help="Output list[list[str]] pkl file.")
    parser.add_argument("--strategy",    required=True,
                        choices=["random", "bm25", "bm25_clusters", "echo"],
                        help="Grouping strategy.")
    parser.add_argument("--num_docs",    required=True, type=int,
                        help="Number of context documents to concatenate into group[0].")
    parser.add_argument("--n_clusters",  type=int, default=None,
                        help="Number of clusters (bm25_clusters strategy only).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.strategy == "bm25_clusters" and args.n_clusters is None:
        parser.error("--n_clusters is required for the bm25_clusters strategy.")

    _random.seed(args.seed)

    docs = load_pkl(args.source_pkl)
    print(f"Loaded {len(docs)} documents from {args.source_pkl}.")

    if args.strategy == "random":
        groups = group_random(docs, args.num_docs)
    elif args.strategy == "bm25":
        groups = group_bm25(docs, args.num_docs)
    elif args.strategy == "bm25_clusters":
        groups = group_bm25_clusters(docs, args.num_docs, args.n_clusters)
    else:
        groups = group_echo(docs, args.num_docs)

    save_pkl(args.dest_pkl, groups)
    print(f"Saved {len(groups)} groups to {args.dest_pkl}.")


if __name__ == "__main__":
    main()
