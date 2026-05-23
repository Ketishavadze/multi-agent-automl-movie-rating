import os
import re

from config import (
    ENGINEERED_DATA_PATH,
    MODEL_LOG_PATH,
    FINAL_REPORT_PATH,
    TARGET_COLUMN,
)

from tools.code_executor import execute_python_code
from llm_client import ask_llm


def clean_llm_code(code_text):
    """
    Removes markdown fences and removes deprecated XGBoost parameters.
    """
    code_text = code_text.strip()

    if code_text.startswith("```"):
        code_text = re.sub(r"^```python", "", code_text)
        code_text = re.sub(r"^```", "", code_text)
        code_text = re.sub(r"```$", "", code_text)

    # Remove deprecated XGBoost parameter if the LLM adds it
    code_text = code_text.replace("use_label_encoder=False,", "")
    code_text = code_text.replace(", use_label_encoder=False", "")
    code_text = code_text.replace("use_label_encoder=False", "")

    return code_text.strip()

def run_model_trainer():
    logs = []
    logs.append("Agent 3 — Model Trainer")
    logs.append("Goal: Generate and execute Python code to train an XGBoost model.")
    logs.append("The agent must inspect results and decide whether to improve or stop.\n")

    engineered_path = os.path.abspath(ENGINEERED_DATA_PATH)

    previous_results = "No previous attempts yet."

    for attempt in range(1, 4):
        logs.append(f"\n================ ATTEMPT {attempt} ================\n")

        system_prompt = """
You are Agent 3: The Model Trainer.

You generate executable Python code for training an XGBoost classification model.

- Return ONLY Python code.
- Do not use markdown.
- Do not explain outside the code.
- The code must load the engineered dataset from the given CSV path.
- The target column is provided.
- Use train_test_split.
- Use XGBClassifier.
- Do not include use_label_encoder in XGBClassifier.
- Do not write use_label_encoder=False.
- This parameter is deprecated and creates warnings.
- Print accuracy, recall, and F1 clearly.
- Print metrics in this exact format:
  accuracy=...
  recall=...
  f1=...
- If this is not the first attempt, adjust hyperparameters based on previous results.
"""

        user_prompt = f"""
Engineered dataset path:
{engineered_path}

Target column:
{TARGET_COLUMN}

Attempt number:
{attempt}

Previous execution results:
{previous_results}

Generate Python code to train and evaluate an XGBoost classifier.
"""

        try:
            code = ask_llm(system_prompt, user_prompt)
            code = clean_llm_code(code)

            logs.append("LLM generated this training code:\n")
            logs.append(code)
            logs.append("\nExecuting generated code...\n")

            result = execute_python_code(code)

            stdout = result["stdout"]
            stderr = result["stderr"]

            logs.append("STDOUT:")
            logs.append(stdout)

            logs.append("STDERR:")
            logs.append(stderr)

            previous_results = f"""
Attempt {attempt} stdout:
{stdout}

Attempt {attempt} stderr:
{stderr}
"""

        except Exception as e:
            logs.append(f"Attempt {attempt} failed before execution: {e}")
            previous_results = f"Attempt {attempt} failed before execution: {e}"
            continue

        # Ask LLM whether model is good enough
        decision_system_prompt = """
You are Agent 3: The Model Trainer.

You must decide whether the current model is good enough or whether another attempt is needed.

Return your answer in this exact format:

DECISION: STOP

or

DECISION: CONTINUE

Then give one short reason.
"""

        decision_user_prompt = f"""
Attempt number:
{attempt}

Execution result:
{previous_results}

Decide whether to stop or continue.

Guidelines:
- Stop if F1 and recall look acceptable or if performance improved enough.
- Continue if there was an error.
- Continue if recall or F1 is weak and another hyperparameter attempt may improve it.
- Maximum attempts is 3.
- If this is attempt 3, choose STOP.
"""

        try:
            decision = ask_llm(decision_system_prompt, decision_user_prompt)

            logs.append("\nLLM Decision:")
            logs.append(decision)

            if "DECISION: STOP" in decision.upper() or attempt == 3:
                logs.append("\nFinal decision: stopping model training loop.")
                break

        except Exception as e:
            logs.append(f"Decision step failed: {e}")
            if attempt == 3:
                break

    final_logs = "\n".join(logs)

    with open(MODEL_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(final_logs)

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Multi-Agent AutoML Team for Movie Rating Classification\n\n")
        f.write("## Agent 3: Model Trainer Logs\n\n")
        f.write("```text\n")
        f.write(final_logs)
        f.write("\n```\n")

    return final_logs