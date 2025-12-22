from __future__ import annotations

from pathlib import Path

import networkx as nx

from graph2emb import Node2Vec
from graph2emb.edges import HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 작은 그래프로 예제 실행 (빠르게)
    g = nx.Graph()
    g.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "A"),
            ("A", "C"),
        ]
    )

    node2vec = Node2Vec(
        g,
        dimensions=8,
        walk_length=4,
        num_walks=3,
        workers=1,
        quiet=True,
        seed=42,
    )
    model = node2vec.fit(window=2, min_count=1, epochs=1)

    # edge embedding (임의의 두 노드 쌍에 대해 계산 가능)
    edge = ("A", "B")
    had = HadamardEmbedder(model.wv, quiet=True)[edge]
    avg = AverageEmbedder(model.wv, quiet=True)[edge]
    l1 = WeightedL1Embedder(model.wv, quiet=True)[edge]
    l2 = WeightedL2Embedder(model.wv, quiet=True)[edge]

    print("edge:", edge)
    print("hadamard:", had[:5], "...", "dim=", had.shape[0])
    print("average :", avg[:5], "...", "dim=", avg.shape[0])
    print("l1      :", l1[:5], "...", "dim=", l1.shape[0])
    print("l2      :", l2[:5], "...", "dim=", l2.shape[0])

    # 전체 edge space를 KeyedVectors로 만들 수도 있습니다(노드 수가 크면 매우 커질 수 있음)
    edges_kv = HadamardEmbedder(model.wv, quiet=True).as_keyed_vectors()
    edge_emb_path = out_dir / "edge_embeddings.word2vec.txt"
    edges_kv.save_word2vec_format(str(edge_emb_path), binary=False)
    print(f"saved: {edge_emb_path}")


if __name__ == "__main__":
    main()


