# sample

이 폴더에는 `graph2emb`의 **실행 가능한 예제**가 들어있습니다.

## 실행 방법

프로젝트 루트에서 아래처럼 실행하세요.

```bash
# 의존성/프로젝트 설치
uv sync

# Node2Vec 예제 (edge list 파일 로드)
uv run python sample/node2vec_from_edgelist.py

# Node2Vec + edge embedding 예제
uv run python sample/node2vec_edge_embeddings.py

# Graph2Vec 예제
uv run python sample/graph2vec_basic.py
```

## 생성 파일

예제 실행 후 결과는 기본적으로 `sample/out/` 아래에 저장됩니다.

## 실행이 너무 오래 걸리면?

이 폴더의 예제들은 **빠르게 확인**할 수 있도록 파라미터를 작게 잡았습니다.
- `Node2Vec`: `dimensions`, `walk_length`, `num_walks`, `epochs`가 커질수록 느려집니다.
- `Graph2Vec`: `dimensions`, `epochs`, (그래프 개수/크기)가 커질수록 느려집니다.


