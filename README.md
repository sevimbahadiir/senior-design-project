# Digital Twin-Based Auto-Scaling Optimization for Microservices
### An LLM-Assisted Multi-Agent Reinforcement Learning Approach

> **TÜBİTAK 2209-A** — University Students Research Projects Support Programme  
> Istanbul Bilgi University, Department of Computer Engineering

---

## Overview

This repository contains the full pipeline for our senior design project, in which we built an autonomous pod scaling framework for Kubernetes-based microservice applications. The system combines three components:

- **Digital Twin** — an MLP-based simulator trained on real cluster telemetry, used as a safe offline training environment for RL agents
- **IPPO-MARL** — six independent PPO agents, one per microservice, learning proactive scaling policies inside the Digital Twin
- **LLM-based XAI** — an explainability layer that automatically diagnoses service-level scaling behavior and generates human-readable explanations

> **Note on the project name:** The original title included "LLM-Based" in the decision-making context. During development, we moved the LLM out of the control loop entirely — scaling decisions are made by MARL agents, and the LLM is used only for post-hoc explanation in the XAI layer.

The framework was evaluated on [Stan's Robot Shop](https://github.com/instana/robot-shop) deployed on a 3-node Kind (Kubernetes-in-Docker) cluster. Compared to the Kubernetes HPA baseline, IPPO-MARL achieves **5× higher cumulative reward** and reduces user service latency by **78.4%**.

---

## Repository Structure

```
senior-design-project/
│
├── 1 - collect data/       # Locust load generation + Prometheus metric collection
├── 2 - sar/                # SAR (State-Action-Reward) transformation & preprocessing
├── 3 - digital-twin/       # Digital Twin MLP training & fidelity evaluation
├── 4 - marl/               # IPPO agent, environment, training & online fine-tuning
└── 5 - XAI/                # 4-step XAI pipeline: surrogate → diagnosis → LLM verbalization
```

---

## Pipeline

```
Kubernetes Cluster (Robot Shop)
        ↓  Prometheus metrics (10s interval)
1. Data Collection & Preprocessing
        ↓  18,708 SAR transitions
2. Digital Twin Training  (R² = 0.886)
        ↓  Safe simulation environment
3. IPPO-MARL Offline Training  (10,000 episodes)
        ↓  Learned policy
4. Online Fine-Tuning on Live Cluster
        ↓  Every 20 steps
5. XAI Pipeline  (diagnosis + LLM explanation)
```

---

## Module Details

### `1 - collect data/`
Locust-based load generation scripts covering 13 workload scenarios (10 standard + 3 targeted low-pod runs). Metrics collected via Istio + Prometheus for 6 services: cart, catalogue, shipping, ratings, user, payment.

### `2 - sar/`
Transforms raw telemetry into State-Action-Reward quadruples `(s_t, a_t, r_t, s_{t+1})`. Includes the tanh-based reward function with per-service criticality weights and MinMax normalization (fit on training partition only).

### `3 - digital-twin/`
MLP transition model (`R^{36} → 256 → 256 → 128 → R^{30}`) trained with Huber loss and early stopping. Achieves R² = 0.886, MAE = 0.042, RMSE = 0.091 on the normalized state space.

### `4 - marl/`
- `ippo_agent.py` — Actor-Critic networks ([128, 64], Tanh), entropy floor mechanism (η_min = 0.3)
- `marl_env.py` — Digital Twin environment wrapper with partial observability
- `train_ippo_v5.py` — Offline training (10,000 episodes)
- `rl_scaler_online.py` — Online fine-tuning on live cluster with automatic XAI triggering

### `5 - XAI/`
Four-step automated pipeline triggered every 20 fine-tuning steps:
1. **Behavior dataset** — 200 DT episodes × 10 steps = 2,000 transitions/agent
2. **Surrogate model** — Gradient Boosting classifier (n_estimators=100, max_depth=4) + variance fallback for near-deterministic agents
3. **Diagnosis engine** — Rule-based classifier: MARL Optimized / Healthy / Under-Provisioned / Pod-Insensitive Bottleneck / Reward Excluded
4. **LLM verbalization** — OpenRouter API (template fallback when unavailable)

---

## Environment & Requirements

**Hardware used:** Windows host, AMD Ryzen 5 5600, 64 GB RAM  
**Cluster:** 3-node Kind (Kubernetes-in-Docker), Istio service mesh

```bash
pip install torch numpy pandas scikit-learn matplotlib
pip install locust kubernetes prometheus-api-client
```

For XAI LLM verbalization, set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=your_key_here
```

---

## What's Not Included

To keep the repository lightweight, the following are not uploaded:
- Raw and preprocessed datasets (`rl_dataset_sar_normalized.csv`)
- Trained model checkpoints (`digital_twin_best.pth`, `ippo_v5_final.pth`)
- XAI output plots and diagnosis reports
