from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from graph2emb import Graph2Vec


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 학습용 그래프들 (작고 빠르게)
    train_graphs = [
        nx.fast_gnp_random_graph(n=8, p=0.3, seed=1),
        nx.fast_gnp_random_graph(n=8, p=0.25, seed=2),
    ]

    g2v = Graph2Vec(
        wl_iterations=2,
        dimensions=16,
        workers=1,
        seed=42,
        min_count=1,
        epochs=1,
    )
    g2v.fit(train_graphs)

    emb = g2v.get_embedding()
    print("train_embedding_shape:", emb.shape)
    print("train_embedding_first_row:", np.round(emb[0][:8], 4))

    # 새로운 그래프들에 대한 추론(infer)
    test_graphs = [nx.fast_gnp_random_graph(n=8, p=0.3, seed=4)]
    inferred = g2v.infer(test_graphs)
    print("infer_embedding_shape:", inferred.shape)
    print("infer_embedding_first_row:", np.round(inferred[0][:8], 4))

    out_path = out_dir / "graph_embeddings.npy"
    np.save(out_path, emb)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()


