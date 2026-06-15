import os
import json
import pandas as pd
import requests


SERVICES = ["cart", "catalogue", "payment", "shipping", "ratings", "user"]

DIAG_LABELS = {
    "pod_insensitive_bottleneck": "Pod-Insensitive Bottleneck",
    "marl_optimized": "MARL Optimized",
    "reward_excluded": "Reward Excluded",
    "under_provisioned": "Under-Provisioned",
    "over_provisioned": "Over-Provisioned",
    "healthy": "Healthy",
    "monitoring_required": "Monitoring Required",
}

OPENROUTER_MODEL = "openai/gpt-oss-120b:free"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_real_feature_importance(service, top_k=5):
    df = pd.read_csv("xai_feature_importance.csv")

    svc_df = (
        df[df["service"] == service]
        .sort_values("importance", ascending=False)
        .head(top_k)
    )

    features = []

    for _, row in svc_df.iterrows():
        features.append(
            {
                "feature": row["feature"],
                "score": round(float(row["importance"]), 3),
                "importance_type": row.get("importance_type", "unknown"),
                "dominant_action": row.get("dominant_action", "unknown"),
            }
        )

    return features


def build_evidence_pack():
    diagnosis_report = load_json("xai_diagnosis_report.json")

    evidence_pack = {}

    for service in SERVICES:
        report = diagnosis_report[service]
        details = report["details"]

        evidence_pack[service] = {
            "service": service,
            "diagnosis": report["diagnosis"],
            "diagnosis_label": DIAG_LABELS.get(
                report["diagnosis"], report["diagnosis"]
            ),
            "severity": report["severity"],
            "action": report["action"],
            "confidence": report["confidence"],
            "latency_ms": details.get("latency_ms"),
            "p50_ms": details.get("p50_ms"),
            "latency_ratio": details.get("latency_ratio"),
            "cpu_norm": details.get("cpu_norm"),
            "pod_count": details.get("pod_count"),
            "pct_improvement": details.get("pct_improvement"),
            "rb_lat_ms": details.get("rb_lat_ms"),
            "ippo_lat_ms": details.get("ippo_lat_ms"),
            "rule_evidence": report.get("evidence", []),
            "top_features": load_real_feature_importance(service, top_k=5),
        }

    with open("xai_evidence_pack.json", "w", encoding="utf-8") as f:
        json.dump(evidence_pack, f, indent=2, ensure_ascii=False)

    return evidence_pack


def template_explanation(evidence):
    service = evidence["service"]
    diagnosis = evidence["diagnosis_label"]
    action = evidence["action"]
    confidence = evidence["confidence"]
    latency = evidence["latency_ms"]
    p50 = evidence["p50_ms"]
    ratio = evidence["latency_ratio"]
    improvement = evidence["pct_improvement"]
    top_features = evidence["top_features"]

    top_feature_text = ", ".join([f["feature"] for f in top_features[:3]])

    if evidence["diagnosis"] == "pod_insensitive_bottleneck":
        return (
            f"The {service} service is classified as {diagnosis}. "
            f"The MARL agent selected {action} with a confidence of {confidence:.2f}, "
            f"while latency remained high at {latency:.1f} ms compared with the p50 threshold of {p50:.1f} ms "
            f"({ratio:.2f}x). The most influential feature signals include {top_feature_text}. "
            f"This suggests that increasing pod count alone may not resolve the degradation, "
            f"indicating a likely dependency-related or I/O-bound bottleneck."
        )

    if evidence["diagnosis"] == "marl_optimized":
        return (
            f"The {service} service is classified as {diagnosis}. "
            f"The MARL policy selected {action} with a confidence of {confidence:.2f}. "
            f"Compared with the rule-based baseline, latency improved by {improvement:.1f}%. "
            f"The most influential feature signals include {top_feature_text}. "
            f"This indicates that the learned policy produced an effective scaling behavior for this service."
        )

    if evidence["diagnosis"] == "reward_excluded":
        return (
            f"The {service} service is marked as {diagnosis}. "
            f"It was monitored during evaluation but excluded from the reward function. "
            f"The agent selected {action} with a confidence of {confidence:.2f}, and the observed latency was {latency:.1f} ms. "
            f"This service should therefore be interpreted as contextual telemetry rather than a direct optimization target."
        )

    return (
        f"The {service} service is classified as {diagnosis}. "
        f"The MARL agent selected {action} with a confidence of {confidence:.2f}. "
        f"The observed latency was {latency:.1f} ms compared with the p50 threshold of {p50:.1f} ms. "
        f"The most influential feature signals include {top_feature_text}."
    )


def build_prompt(evidence):
    top_features = "\n".join(
        [
            f"- {f['feature']} "
            f"(relative score: {f['score']}, type: {f['importance_type']})"
            for f in evidence["top_features"]
        ]
    )

    rule_evidence = "\n".join([f"- {ev}" for ev in evidence["rule_evidence"]])

    return f"""
You are explaining a pre-computed XAI result for a MARL-based microservice autoscaling system.




Service: {evidence["service"]}
Diagnosis: {evidence["diagnosis_label"]}
Severity: {evidence["severity"]}
MARL Action: {evidence["action"]}
Action Confidence: {evidence["confidence"]}

Latency: {evidence["latency_ms"]} ms
p50 Threshold: {evidence["p50_ms"]} ms
Latency Ratio: {evidence["latency_ratio"]}x
CPU Normalized: {evidence["cpu_norm"]}
Pod Count: {evidence["pod_count"]}

Rule-based Evidence:
{rule_evidence}

Top Feature Importance Signals:
{top_features}

Write an evidence-guided explanation suitable for an academic paper.
""".strip()


def call_openrouter(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "XAI MARL Explanation",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful academic technical writer. "
                    "You only interpret provided evidence and never invent new diagnoses."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            print(f"    OpenRouter warning: {response.status_code}")
            print(f"    Detail: {response.text[:300]}")
            print("    Using template fallback.")
            return None

        data = response.json()
        text = data["choices"][0]["message"]["content"]

        return text.strip() if text.strip() else None

    except Exception as e:
        print(f"    OpenRouter API error: {e}")
        print("    Using template fallback.")
        return None


def main():
    print("=" * 70)
    print("XAI Step 4: Evidence-Guided LLM Explanation")
    print("=" * 70)

    evidence_pack = build_evidence_pack()
    explanations = {}

    use_api = os.getenv("OPENROUTER_API_KEY") is not None
    print(
        f"\nOpenRouter API status: "
        f"{'ACTIVE' if use_api else 'NONE -- using template fallback'}\n"
    )

    for service, evidence in evidence_pack.items():
        print(f"[{service}] generating explanation...")

        prompt = build_prompt(evidence)
        llm_text = call_openrouter(prompt)

        if llm_text:
            explanation = llm_text
            method = f"openrouter_{OPENROUTER_MODEL}"
        else:
            explanation = template_explanation(evidence)
            method = "template_fallback"

        explanations[service] = {
            "diagnosis": evidence["diagnosis_label"],
            "severity": evidence["severity"],
            "method": method,
            "explanation": explanation,
            "top_features": evidence["top_features"],
        }

        print(f"  OK method: {method}")
        print(f"  {explanation[:160]}...\n")

    with open("xai_evidence_guided_explanations.json", "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    lines = [
        "=" * 80,
        "XAI EVIDENCE-GUIDED EXPLANATION REPORT",
        "IPPO Policy Analysis — Robot Shop Microservice Autoscaling",
        "=" * 80,
        "",
        "Note:",
        "Diagnoses are generated by the deterministic rule-based engine.",
        "The explanation layer only interprets pre-computed evidence such as diagnosis, latency, confidence, and feature importance signals.",
        f"LLM provider/model: OpenRouter / {OPENROUTER_MODEL if use_api else 'template fallback'}",
        "",
    ]

    icon_map = {
        "ok": "[OK]",
        "info": "[INFO]",
        "warning": "[WARN]",
        "critical": "[CRIT]",
    }

    for service, item in explanations.items():
        icon = icon_map.get(item["severity"], "[?]")
        lines.extend(
            [
                f"{icon} {service.upper()} — {item['diagnosis']}",
                "-" * 60,
                f"Method: {item['method']}",
                "",
                item["explanation"],
                "",
            ]
        )

    lines.extend(
        [
            "=" * 80,
            "Generated by: Evidence-guided XAI verbalization layer",
            "=" * 80,
        ]
    )

    with open("xai_evidence_guided_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Outputs:")
    print("  OK xai_evidence_pack.json")
    print("  OK xai_evidence_guided_explanations.json")
    print("  OK xai_evidence_guided_report.txt")
    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
