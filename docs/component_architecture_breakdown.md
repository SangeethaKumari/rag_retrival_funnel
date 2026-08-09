# Retrieval Funnel: Component Architecture Breakdown

This document provides a component-by-component architectural breakdown of the retrieval funnel pipeline implemented in this project.

---

## 📋 Stage 0: The Retrieval Funnel Big Picture

The retrieval funnel chains different models in order of **increasing per-document cost and accuracy**. By executing nested server-side Qdrant `prefetch` queries, it retrieves the bulk of the corpus cheaply, merges semantic and lexical search paths, and performs high-fidelity ranking at the very end.

```mermaid
flowchart TD
    subgraph CORPUS["Indexed corpus (all chunks: Biology · Physics · History)"]
        direction TB
        Z[" "]
    end

    CORPUS ==>|"scan everything, cheaply"| S1

    S1["🔵 Stage 1 — Matryoshka 64-d ANN search<br/><b>500 candidates</b><br/>cost per doc: tiny (64-d cosine, HNSW)"]
    S2["🔵 Stage 2 — Matryoshka 768-d rescoring<br/><b>100 candidates</b><br/>cost per doc: small (768-d cosine, only on 500)"]
    S3["🟢 Parallel — SPLADE sparse retrieval<br/><b>100 candidates</b><br/>cost per doc: tiny (inverted-index dot product)"]
    S4["🟣 Stage 3 — Merge + ColBERT MaxSim<br/><b>50 candidates</b><br/>cost per doc: moderate (token-level late interaction)"]
    S5["🔴 Stage 4 — Cross-encoder rerank<br/><b>20 final results</b><br/>cost per doc: high (full transformer pass per pair)"]

    S1 --> S2
    CORPUS ==>|"keyword branch"| S3
    S2 --> S4
    S3 --> S4
    S4 --> S5
    S5 --> OUT(["Top-20 precisely ranked results"])

    style CORPUS fill:#eef3fa,stroke:#4a6fa5
    style S1 fill:#dbe7f5,stroke:#4a6fa5
    style S2 fill:#c3d7ee,stroke:#4a6fa5
    style S3 fill:#e2eed3,stroke:#6b8f3e
    style S4 fill:#e3d7f0,stroke:#7a5aa8
    style S5 fill:#f5d9d4,stroke:#b0553f
    style OUT fill:#fdf3d8,stroke:#c19a2e
```

### Key Funnel Design Principles:
1. **Resource Conservation**: We avoid passing all documents through expensive models (like Cross-Encoders or ColBERT).
2. **Parallel Hybrid Recall**: Dense (Matryoshka) and Sparse (SPLADE) retrieval run in parallel to capture both conceptual semantics and exact terms.
3. **Early Stage Recall Priority**: Early stages focus strictly on high recall, while later stages optimize for precision.

---

## 🧸 Component 1: Matryoshka Representation Learning (MRL)

Matryoshka embeddings are trained so that their leading dimensions carry the coarsest, most important semantic structure, while subsequent dimensions add finer details. This allows us to perform a very fast Approximate Nearest Neighbor (ANN) search over small 64-d vectors, and then refine only the top 500 candidates using the full 768-d vectors.

### Vector Structure & Trait Aggregation
```mermaid
flowchart LR
    subgraph V["One 768-d Matryoshka embedding"]
        direction LR
        A["dims 1–64<br/>coarse topical<br/>structure"] --- B["dims 65–256<br/>finer semantic<br/>distinctions"] --- C["dims 257–768<br/>fine-grained<br/>nuance"]
    end
    A -->|"prefix = usable<br/>64-d embedding"| U1(["fast, approximate<br/>similarity"])
    V -->|"full vector"| U2(["precise<br/>similarity"])

    style A fill:#c3d7ee,stroke:#4a6fa5
    style B fill:#dbe7f5,stroke:#4a6fa5
    style C fill:#eef3fa,stroke:#4a6fa5
```

### The Query-Time Cascade
```mermaid
flowchart TD
    Q["query"] --> QE["encode once → 768-d vector"]
    QE --> Q64["slice to 64-d"]
    QE --> Q768["keep full 768-d"]

    subgraph STAGE1["Stage 1 — shortlist (cheap)"]
        Q64 --> ANN["ANN search over 64-d index<br/>~12× less memory traffic,<br/>~12× cheaper distances"]
        ANN --> C500["top 500 candidates"]
    end

    subgraph STAGE2["Stage 2 — refine (precise)"]
        Q768 --> RS["exact 768-d cosine rescoring<br/>of only those 500"]
        C500 --> RS
        RS --> C100["top 100, near-full-precision ranking"]
    end

    style STAGE1 fill:#eef3fa,stroke:#4a6fa5
    style STAGE2 fill:#e2eed3,stroke:#6b8f3e
```

### Why it works:
During training, the loss function is evaluated simultaneously on multiple prefixes (`u[1:64]`, `u[1:128]`, `u[1:768]`). This forces the encoder to package primary information (like subject classifications) in the first few dimensions.

---

## 🔍 Component 2: SPLADE Sparse Lexical Retrieval

SPLADE yields **sparse vectors** across a 30,000 WordPiece vocabulary. Unlike BM25 which computes static frequency scores, SPLADE leverages a transformer MLM head to compute contextual weights and perform **term expansion**, inserting synonyms and related words not found in the original text.

### Inverted Index & Term Expansion
```mermaid
flowchart LR
    T["'How do vaccines work<br/>in the human body?'"] --> M["SPLADE encoder<br/>(BERT + MLM head)"]
    M --> V["sparse vector over vocabulary"]
    subgraph V2["non-zero entries (illustrative)"]
        direction TB
        w1["vaccine : 2.9"]
        w2["immune : 2.1"]
        w3["body : 1.4"]
        w4["antibody : 1.2  ← expansion"]
        w5["dose : 0.7  ← expansion"]
        w6["how : 0.1"]
    end
    V --- V2
    V2 --> IDX[("inverted index /<br/>Qdrant sparse vectors")]

    style M fill:#e3d7f0,stroke:#7a5aa8
    style V2 fill:#e2eed3,stroke:#6b8f3e
    style IDX fill:#fdf3d8,stroke:#c19a2e
```

### Training & FLOPS Regularization
```mermaid
flowchart TD
    D["(query, positive doc, hard negatives)"] --> S["SPLADE encoder"]
    S --> R["ranking loss<br/>(+ distillation from cross-encoder)"]
    S --> F["FLOPS regularizer<br/>(penalize non-sparse vectors)"]
    R & F --> L["total loss → backprop"]

    style S fill:#e3d7f0,stroke:#7a5aa8
    style R fill:#f5d9d4,stroke:#b0553f
    style F fill:#e2eed3,stroke:#6b8f3e
```

### Why SPLADE is critical:
It acts as a safety net. While dense embeddings capture broad conceptual overlap, SPLADE handles exact keywords, numbers, jargon, and proper nouns (like "Newton's second law" or "Treaty of Westphalia") that are easily blurred by dense representations.

---

## 🤝 Component 3: ColBERT Late Interaction

ColBERT sits in the sweet spot between bi-encoders and cross-encoders. Instead of summarizing the entire passage into a single vector, it produces a 128-d vector **for every token**. Interaction is deferred to query-time (hence "late interaction") via a cheap `MaxSim` calculation.

### Late Interaction vs. Alternatives
```mermaid
flowchart LR
    subgraph BI["Bi-encoder (early aggregation)"]
        direction TB
        bq["query → 1 vector"] --- bd["document → 1 vector"]
        bq -.->|"one dot product"| bs["score"]
        bd -.-> bs
    end
    subgraph CO["ColBERT (late interaction)"]
        direction TB
        cq["query → vector per token"] --- cd["document → vector per token"]
        cq -.->|"MaxSim over token pairs"| cs["score"]
        cd -.-> cs
    end
    subgraph CR["Cross-encoder (full interaction)"]
        direction TB
        xq["query ⧺ document<br/>read jointly"] --> xs["score"]
    end
    BI -->|"more accurate"| CO -->|"more accurate"| CR
    CR -->|"cheaper"| CO -->|"cheaper"| BI

    style BI fill:#eef3fa,stroke:#4a6fa5
    style CO fill:#e3d7f0,stroke:#7a5aa8
    style CR fill:#f5d9d4,stroke:#b0553f
```

### Token-level MaxSim Matching
```mermaid
flowchart TD
    subgraph QT["Query token vectors"]
        q1["force"]
        q2["acceleration"]
        q3["relationship"]
    end
    subgraph DT["Document token vectors"]
        d1["Newton"]
        d2["second"]
        d3["law"]
        d4["F = ma"]
        d5["proportional"]
        d6["mass"]
    end
    q1 -->|"max-sim → 'F = ma'"| d4
    q2 -->|"max-sim → 'proportional'"| d5
    q3 -->|"max-sim → 'law'"| d3
    q1 & q2 & q3 -.-> SUM["Σ of the three best matches = S(Q, D)"]

    style QT fill:#eef3fa,stroke:#4a6fa5
    style DT fill:#f4f7f4,stroke:#6b8f3e
    style SUM fill:#fdf3d8,stroke:#c19a2e
```

### Why it works:
Because token vectors are contextualized by the transformer prior to projection, the term vectors carry semantic meaning based on context (e.g. "bank" near "river" has a different representation than "bank" near "money"). Document tokens can be precomputed and stored, which is why late-interaction is so efficient compared to Cross-Encoders.
