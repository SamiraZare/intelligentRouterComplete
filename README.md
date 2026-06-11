# 🤖 Intelligent AI Router
### Unsupervised Learning · Anomaly Detection · Recommender Systems · Reinforcement Learning + Runtime Serving Layer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Course](https://img.shields.io/badge/DeepLearning.AI-Course%203-orange)](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning)
[![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

A query router that decides **how** an AI system should answer each query — direct LLM, retrieval (RAG), cache, or external tool — by combining four machine-learning components from the DeepLearning.AI *Unsupervised Learning, Recommenders & RL* course, then **serves the trained policy through a runnable FastAPI service**.

This is not a notebook. It trains a full stack of models, saves them as artifacts, and loads them into a live routing service you can query from the command line or over HTTP.

---

## 📊 Results (actual, from `python src/train.py` on real data)

**Dataset: 1,161 real questions** — 874 AI2 NDMC science exam questions (the New York Regents pool that the ARC benchmark was built from, fetched from the official [allenai/aristo-mini](https://github.com/allenai/aristo-mini) repo) + 300 GSM8K math word problems (from the official [openai/grade-school-math](https://github.com/openai/grade-school-math) repo). 870 train / 291 test, one split per unique question — no leakage.

| System | Composite Score | Routing Accuracy |
|---|---|---|
| Majority baseline | 0.213 | 33.7% |
| Neural router (original binary policy) | 0.315 | 54.0% |
| CF recommender | 0.046 | 43.3% |
| KNN recommender | 0.281 | 54.3% |
| **Full system (KNN + CF + RL)** | **0.343** | **65.3%** |

**The full system wins on both metrics**: +9% relative composite over the binary neural-router policy and an 11-point routing-accuracy gain (65.3% vs 54.0/54.3%).

Full-system operating profile: average expected accuracy **0.755**, average cost **$0.072/query**, average latency **0.475s**, average hallucination risk **0.212**.

![System comparison](figures/composite_comparison.png)

> **Why composite, not raw accuracy?** Composite = accuracy − 0.30·cost − 0.20·latency − 0.50·hallucination. A route that is slightly less accurate but far cheaper and safer can be the right production choice. The neural router optimised one binary decision; this system optimises the full tradeoff.

---

## 🧩 The four ML components (Course 3)

| Topic | File | Role |
|---|---|---|
| **K-Means clustering** | `feature_engineering.py` | Clusters TF-IDF query vectors into semantic topic groups |
| **Anomaly detection** | `anomaly_detector.py` | Fits Gaussian p(x); flags novel queries (log p(x) < ε) that should not be cached |
| **Collaborative filtering** | `collaborative_filter.py` | Matrix factorization (gradient descent from scratch) recommends a route per cluster |
| **KNN recommender** | `recommender.py` | Votes on the best route from the 7 most similar past queries |
| **Q-learning** | `q_learning_agent.py` + `environment.py` | Learns the routing policy; on test data, 284/291 decisions come from the learned Q-table, and it overrides its KNN input on 117/291 queries |

---

## 🏗 Architecture

```
TRAINING (src/train.py)                      SERVING (src/router_service.py)
─────────────────────────                    ──────────────────────────────
ARC-like dataset                             Incoming query
   │                                            │
TF-IDF + K-Means          ┌── saves ──┐         ├─ exact-duplicate cache check
   │                      │ artifacts │         ├─ TF-IDF → cluster, anomaly bin
Gaussian anomaly detector │  *.pkl    │ ──────► ├─ KNN + CF recommendations
   │                      │  *.json   │         ├─ RL agent picks route
Oracle labels (composite) └───────────┘         ├─ calculator guard (exact math only)
   │                                            └─ execute route → answer + trace
KNN recommender
   │
Collaborative filter
   │
Q-learning agent  ◄── reward = composite (accuracy − cost − latency − hallucination)
```

The serving layer applies exactly **one** deterministic guard (route provably-arithmetic queries to the calculator). Every other decision — RAG vs direct vs cache — is the **learned RL policy**, not a hand-written rule.

---

## 🚀 Quickstart

```bash
git clone https://github.com/SamiraZare/intelligent-ai-router.git
cd intelligent-ai-router
pip install -r requirements.txt

# 1. Build the real dataset from official GitHub sources (one-time)
python src/download_data.py

# 2. Train everything (~60s on CPU) — writes artifacts/, results/, figures/
python src/train.py

# 3. Route a query through the trained policy
python src/demo.py --query "What is 45 + 17?"
python src/demo.py --query "Which planet is known as the Red Planet?"

# 4. Or serve over HTTP
uvicorn src.app:app --reload
# POST {"query": "..."} to http://localhost:8000/route
```

### Example runtime outputs (real)

```
$ python src/demo.py --query "Janet has 16 eggs. Calculate 16 * 2."
  route: external_tool      answer: The result of 16*2 is 32.

$ python src/demo.py --query "Which stage comes next in the life cycle of a bird after hatching?"
  route: rag_retrieval      rl_action: rag_retrieval
```

---

## 📁 Project structure

```
intelligent-ai-router/
├── src/
│   ├── data_loader.py           # Loads real_dataset.csv (AI2 NDMC + GSM8K)
│   ├── feature_engineering.py   # TF-IDF + K-Means + compact RL state encoding
│   ├── routing_oracle.py        # Per-query per-action accuracy from published benchmarks
│   ├── anomaly_detector.py      # Gaussian anomaly detection
│   ├── recommender.py           # KNN query→action recommender (batched)
│   ├── collaborative_filter.py  # Matrix factorization CF (gradient descent from scratch)
│   ├── environment.py           # RL environment (composite reward)
│   ├── q_learning_agent.py      # Tabular Q-learning (γ=0 contextual bandit + unseen-state fallback)
│   ├── train.py                 # Full training pipeline → saves all artifacts
│   ├── evaluate.py              # 5-way composite + routing-accuracy comparison
│   ├── plot_results.py          # Training curves + comparison chart
│   │
│   ├── router_service.py        # Loads artifacts; routes live queries
│   ├── llm_backend.py           # Offline LLM stand-in (template answers)
│   ├── retriever.py             # TF-IDF retriever over the knowledge base
│   ├── knowledge_base.py        # Builds Q→A docs from the dataset
│   ├── cache_store.py           # Similarity-based runtime cache
│   ├── tool_executor.py         # Safe calculator for arithmetic queries
│   ├── runtime_features.py      # Builds feature row for a live query
│   ├── app.py                   # FastAPI service (/route, /health)
│   └── demo.py                  # CLI entry point
│
├── data/real_dataset.csv        # 1,161 REAL questions: AI2 NDMC science exams + GSM8K
├── src/download_data.py         # Builds it from official allenai/ + openai/ GitHub repos
├── artifacts/                   # Saved models (created by train.py)
├── results/                     # metrics.json, predictions, training history
├── figures/                     # 3 generated charts
├── requirements.txt
└── README.md
```

---

## 🔬 Key design decisions

**Composite-aligned oracle.** `ideal_action` is the action that maximises the *composite* score, not raw accuracy — because the agent is rewarded on composite. Defining the target on a different metric than the reward (a bug in an earlier version) makes every comparison incoherent.

**γ = 0 (contextual bandit).** Each routing decision is independent; there is no state carried between queries. γ=0 is the theoretically correct formulation. The agent observes only the reward for the action it actually takes — no oracle pre-seeding of counterfactual rewards.

**Unseen-state fallback.** With a multi-field discrete state and a few hundred training rows, some test states are never seen in training. Rather than defaulting to action 0, the agent falls back to its KNN recommendation for unseen states. On the test set this happens only 7/291 times — the other 284 use the learned Q-table.

**One deterministic guard, not a rule engine.** The serving layer overrides the RL policy only to route exact arithmetic to the calculator (provably correct). All other routing is the learned policy. An earlier version replaced the RL decision with a chain of if/else rules, which would make the ML decorative — that has been removed.

**No data leakage.** The dataset assigns one split per unique question. The 332 questions are distinct; the test set shares no rows with train. (An earlier sample repeated each question across all three splits, which made the test set a copy of the training set.)

---

## 💬 FAQ

**Q: Why is the CF recommender so much weaker than KNN?**
> CF recommends one action per K-Means cluster — a coarse, cluster-level signal — while KNN looks at the 7 most similar individual queries. On this data the clusters mix question types, so CF's per-cluster majority is often wrong. The interesting result is that the RL agent learns this from reward alone: it leans on KNN where KNN is reliable and ignores CF where it isn't.

**Q: How do I know the RL agent is actually doing something, not just echoing the recommender?**
> On the test set, the full system diverges from its KNN input on 117 of 291 queries, and 284 of 291 decisions come from the learned Q-table rather than the fallback. The gains over KNN (54.3% → 65.3% routing accuracy, 0.281 → 0.343 composite) come precisely from those overrides.

**Q: Why tabular Q-learning instead of DQN?**
> The state space is small and discrete. A lookup table converges reliably without the variance of neural approximation. DQN would be over-engineering here.

**Q: Where do the accuracy numbers come from?**
> Published benchmarks: GPT-3.5-level ARC accuracy (Brown et al. 2020), RAG improvement margins (Lewis et al. 2020), and runtime cosine similarity for cache hit rates. Not hand-invented labels.

---

## 📦 Dependencies

`numpy`, `pandas`, `scikit-learn`, `matplotlib` for the ML pipeline; `fastapi` + `uvicorn` + `pydantic` for serving. No GPU, no PyTorch/TensorFlow. Install with `pip install -r requirements.txt`.

---

## 👩‍💻 Author

**Samira Zare** — ML / AI Engineer | PhD, Computer Engineering, UC Santa Cruz
[samirazare.com](https://www.samirazare.com) · [GitHub](https://github.com/SamiraZare) · samiraaa.zare@gmail.com · Sunnyvale, CA

*Built as an extension of the Neural Retrieval Router, demonstrating every major topic from the DeepLearning.AI Unsupervised Learning, Recommenders, and Reinforcement Learning course (Stanford / Coursera).*

---

## 📄 Resume bullet

> Built an end-to-end AI query router on 1,161 real questions (AI2 Regents science exams + GSM8K) combining K-Means clustering, Gaussian anomaly detection, KNN and matrix-factorization recommenders, and tabular Q-learning, served through a FastAPI service; the learned policy beats the binary neural-router baseline on both composite score (+9% relative) and routing accuracy (65.3% vs 54.0%), with reward signals grounded in published benchmarks (Brown 2020, Lewis 2020, Cobbe 2021, Gao 2022).

## 📜 License

MIT
