from __future__ import annotations

from pathlib import Path

import networkx as nx

from graph2emb import Node2Vec


def load_weighted_edgelist(path: Path) -> nx.Graph:
    g = nx.Graph()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        src, dst = parts[0], parts[1]
        weight = float(parts[2]) if len(parts) >= 3 else 1.0
        g.add_edge(src, dst, weight=weight)
    return g


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "edgelist.txt"
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    g = load_weighted_edgelist(data_path)
    print(f"loaded_graph: nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")

    node2vec = Node2Vec(
        g,
        # sample은 "빠르게 확인" 목적이라 파라미터를 작게 둡니다.
        dimensions=8,
        walk_length=4,
        num_walks=2,
        workers=1,
        quiet=True,
        weight_key="weight",
        seed=42,
    )

    model = node2vec.fit(window=3, min_count=1, epochs=1)

    # 노드 이름은 항상 문자열입니다.
    print("most_similar('2'):", model.wv.most_similar("2", topn=5))

    emb_path = out_dir / "node_embeddings.word2vec.txt"
    model.wv.save_word2vec_format(str(emb_path), binary=False)
    print(f"saved: {emb_path}")


if __name__ == "__main__":
    main()


