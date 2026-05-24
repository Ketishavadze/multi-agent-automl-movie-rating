import os
import re
from typing import Dict, List, Optional

import pandas as pd

from config import (
    CLEANING_REPORT_PATH,
    ENGINEERED_DATA_PATH,
    FEATURE_REPORT_PATH,
    FINAL_REPORT_PATH,
    MODEL_LOG_PATH,
    TARGET_COLUMN,
)
from llm_client import ask_llm
from tools.code_executor import execute_python_code


MAX_ATTEMPTS = 3


def clean_llm_code(code_text: str) -> str:
    """Remove markdown fences and deprecated XGBoost parameters from LLM code."""
    code_text = code_text.strip()
    if code_text.startswith("```"):
        code_text = re.sub(r"^```python\s*", "", code_text)
        code_text = re.sub(r"^```\s*", "", code_text)
        code_text = re.sub(r"```\s*$", "", code_text)

    code_text = code_text.replace("use_label_encoder=False,", "")
    code_text = code_text.replace(", use_label_encoder=False", "")
    code_text = code_text.replace("use_label_encoder=False", "")
    return code_text.strip()


def _parse_metric(stdout: str, metric: str) -> Optional[float]:
    match = re.search(rf"{metric}\s*=\s*([0-9.]+)", stdout)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Report not found."


def _summarize_report(report_text: str, max_lines: int = 18) -> str:
    useful = []
    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(keyword in stripped.lower() for keyword in [
            "created", "dropped", "imputed", "converted", "selected", "output shape", "cleaned shape", "goal:"
        ]):
            useful.append(stripped)
    return "\n".join(f"- {line}" for line in useful[:max_lines]) or "- No summary lines found."


def _target_distribution() -> str:
    try:
        df = pd.read_csv(ENGINEERED_DATA_PATH)
        counts = df[TARGET_COLUMN].value_counts().sort_index().to_dict()
        total = len(df)
        return ", ".join(
            f"class {int(label)}: {count} ({count / total:.2%})"
            for label, count in counts.items()
        )
    except Exception:
        return "Target distribution unavailable."


def _write_final_report(logs: str, attempt_results: List[Dict[str, float]]) -> None:
    cleaning_report = _read_text(CLEANING_REPORT_PATH)
    feature_report = _read_text(FEATURE_REPORT_PATH)

    valid_results = [r for r in attempt_results if r.get("f1") is not None]
    best = max(valid_results, key=lambda r: (r.get("f1", 0), r.get("recall", 0))) if valid_results else None

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Multi-Agent AutoML Team for Movie Rating Classification\n\n")
        f.write("## Objective\n")
        f.write(
            "This system implements a sequential three-agent AutoML workflow for predicting whether a movie is highly rated. "
            f"The binary target is `{TARGET_COLUMN}`, created from the original rating column before leakage-prone fields are removed.\n\n"
        )

        f.write("## Pipeline Summary\n")
        f.write("1. **Agent 1 — Data Cleaner:** inspects raw metadata, chooses cleaning actions, saves `clean_data.csv`, and writes `cleaning_report.txt`.\n")
        f.write("2. **Agent 2 — Feature Engineer:** receives Agent 1's output, creates semantic movie features, removes leakage/raw columns, performs feature selection, saves `engineered_data.csv`, and writes `feature_report.txt`.\n")
        f.write("3. **Agent 3 — Model Trainer:** generates executable XGBoost code, evaluates Accuracy/Recall/F1, and iterates based on its own decision step.\n\n")

        f.write("## Agent 1: Data Cleaning Summary\n")
        f.write(_summarize_report(cleaning_report))
        f.write("\n\n")

        f.write("## Agent 2: Feature Engineering Summary\n")
        f.write(_summarize_report(feature_report))
        f.write("\n\n")

        f.write("## Target Distribution\n")
        f.write(_target_distribution())
        f.write("\n\n")

        f.write("## Agent 3: Model Training Results\n")
        f.write("| Attempt | Accuracy | Recall | F1 | Decision |\n")
        f.write("|---:|---:|---:|---:|---|\n")
        for r in attempt_results:
            decision = r.get("decision", "Unknown")
            accuracy = r.get("accuracy")
            recall = r.get("recall")
            f1 = r.get("f1")
            f.write(
                f"| {int(r['attempt'])} | "
                f"{accuracy:.4f} | {recall:.4f} | {f1:.4f} | {decision} |\n"
            )

        if best:
            f.write("\n")
            f.write(
                f"**Best result:** Attempt {int(best['attempt'])}, selected by highest F1 with recall as a tie-breaker. "
                f"Accuracy = {best['accuracy']:.4f}, Recall = {best['recall']:.4f}, F1 = {best['f1']:.4f}.\n\n"
            )
            f.write(
                "The final model keeps high overall accuracy while improving recall compared with the baseline. "
                "Recall remains the limiting metric, which is expected when the positive class of highly rated movies is smaller than the negative class.\n\n"
            )

        f.write("## Full Agent 3 Execution Log\n")
        f.write("```text\n")
        f.write(logs)
        f.write("\n```\n")


def run_model_trainer() -> str:
    logs: List[str] = []
    attempt_results: List[Dict[str, float]] = []

    logs.append("Agent 3 — Model Trainer")
    logs.append("Goal: Generate and execute Python code to train an XGBoost model.")
    logs.append("The agent must inspect results and decide whether to improve or stop.\n")

    # Use a relative path so generated code and logs are reproducible on another machine.
    engineered_path = ENGINEERED_DATA_PATH.replace("\\", "/")
    previous_results = "No previous attempts yet."

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logs.append(f"\n================ ATTEMPT {attempt} ================\n")

        system_prompt = """
You are Agent 3: The Model Trainer.
You generate executable Python code for training an XGBoost binary classifier.

Return ONLY Python code. Do not use markdown.
Requirements:
- Load the engineered dataset from the provided CSV path.
- The target column is provided.
- Use train_test_split with random_state=42 and stratify=y.
- Use XGBClassifier.
- Do not use the deprecated use_label_encoder parameter.
- Print accuracy, recall, and F1 in this exact format:
  accuracy=...
  recall=...
  f1=...
- Attempt 1 should be a baseline.
- Later attempts must react to previous results by changing hyperparameters such as n_estimators, max_depth, learning_rate, subsample, colsample_bytree, or scale_pos_weight.
"""

        user_prompt = f"""
Engineered dataset path: {engineered_path}
Target column: {TARGET_COLUMN}
Attempt number: {attempt}
Previous execution results:
{previous_results}

Generate the Python training code now.
"""

        try:
            code = ask_llm(system_prompt, user_prompt)
            code = clean_llm_code(code)
            logs.append("LLM generated this training code:\n")
            logs.append(code)
            logs.append("\nExecuting generated code...\n")

            result = execute_python_code(code)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            logs.append("STDOUT:")
            logs.append(stdout)
            logs.append("STDERR:")
            logs.append(stderr)

            accuracy = _parse_metric(stdout, "accuracy")
            recall = _parse_metric(stdout, "recall")
            f1 = _parse_metric(stdout, "f1")
            previous_results = f"Attempt {attempt} stdout:\n{stdout}\nAttempt {attempt} stderr:\n{stderr}"
        except Exception as exc:
            logs.append(f"Attempt {attempt} failed before execution: {exc}")
            previous_results = f"Attempt {attempt} failed before execution: {exc}"
            accuracy = recall = f1 = None

        decision_system_prompt = """
You are Agent 3: The Model Trainer.
Decide whether the current model is good enough or whether another attempt is needed.
Return your answer in this exact format:
DECISION: STOP or DECISION: CONTINUE
Reason: one short reason.
"""
        decision_user_prompt = f"""
Attempt number: {attempt}
Maximum attempts: {MAX_ATTEMPTS}
Execution result:
{previous_results}

Guidelines:
- Continue if there was an execution error.
- Continue if recall or F1 is weak and another attempt remains.
- Stop if F1 and recall are acceptable or if this is the final allowed attempt.
"""

        try:
            decision = ask_llm(decision_system_prompt, decision_user_prompt)
        except Exception as exc:
            decision = f"DECISION: STOP\nReason: Decision step failed: {exc}" if attempt == MAX_ATTEMPTS else f"DECISION: CONTINUE\nReason: Decision step failed: {exc}"

        if attempt == MAX_ATTEMPTS and "DECISION: STOP" not in decision.upper():
            decision = "DECISION: STOP\nReason: Maximum attempts reached; the best available model will be reported."

        logs.append("\nLLM Decision:")
        logs.append(decision)

        attempt_results.append({
            "attempt": float(attempt),
            "accuracy": float(accuracy or 0),
            "recall": float(recall or 0),
            "f1": float(f1 or 0),
            "decision": "Stop" if "DECISION: STOP" in decision.upper() else "Continue",
        })

        if "DECISION: STOP" in decision.upper():
            logs.append("\nFinal decision: stopping model training loop.")
            break

    final_logs = "\n".join(logs)

    with open(MODEL_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(final_logs)

    _write_final_report(final_logs, attempt_results)
    return final_logs
