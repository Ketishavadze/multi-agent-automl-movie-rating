import os
import subprocess
import sys
import tempfile
from typing import Dict


def execute_python_code(code_string: str, timeout: int = 180) -> Dict[str, object]:
    """Run LLM-generated Python code in a temporary file and return logs."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code_string)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
