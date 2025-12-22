# graph2emb

> [Elior Cohen의 node2vec](https://github.com/eliorc/node2vec), [graph2vec](https://github.com/benedekrozemberczki/graph2vec)를 참고해 Node2Vec + Graph2Vec 기능을 한 프로젝트로 합친 구현체입니다.

## 특징

- Node2Vec: 랜덤 워크 기반 노드 임베딩
- Edge embedding: (Hadamard/평균/L1/L2) 방식의 엣지 임베딩
- Graph2Vec: WL(Weisfeiler–Lehman) hashing + Doc2Vec 기반 그래프 임베딩
- uv 기반 개발/실행

> 파이썬 모듈 이름은 `graph2emb` 입니다.

## 빠른 시작 (sample 실행)

```bash
# 의존성/프로젝트 설치
uv sync

# Node2Vec (edge list 로드)
uv run python sample/node2vec_from_edgelist.py

# Node2Vec + edge embedding
uv run python sample/node2vec_edge_embeddings.py

# Graph2Vec
uv run python sample/graph2vec_basic.py
```

샘플 설명은 `sample/README.md`를 참고하세요.

## 사용 예시 (코드)

### Node2Vec (노드 임베딩)

```python
import networkx as nx
from graph2emb import Node2Vec

g = nx.fast_gnp_random_graph(n=100, p=0.3, seed=42)
node2vec = Node2Vec(g, dimensions=64, walk_length=30, num_walks=20, workers=1, seed=42)
model = node2vec.fit(window=10, min_count=1, epochs=5)

print(model.wv.most_similar("2", topn=5))  # 노드 id는 문자열로 조회
```

### Edge embedding (엣지 임베딩)

```python
from graph2emb.edges import HadamardEmbedder

edges_embs = HadamardEmbedder(model.wv)
print(edges_embs[("1", "2")])
```

### Graph2Vec (그래프 임베딩)

```python
import networkx as nx
from graph2emb import Graph2Vec

graphs = [
    nx.fast_gnp_random_graph(n=12, p=0.3, seed=1),
    nx.fast_gnp_random_graph(n=14, p=0.2, seed=2),
]

g2v = Graph2Vec(dimensions=32, workers=1, min_count=1, epochs=3, seed=42)
g2v.fit(graphs)
emb = g2v.get_embedding()  # shape: (len(graphs), dimensions)
```

## 개발/테스트

```bash
# 테스트 실행
uv run pytest

# 병렬 테스트
uv run pytest -n auto

# 커버리지 포함 테스트 실행
uv run pytest --cov=graph2emb --cov-report=html
```
