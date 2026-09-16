"""
Run the data-prep scripts in order from the project root.
"""

import os
import subprocess
import sys

SCRIPTS = [
  "scripts/duplicate_handler.py",
  "scripts/outlier_handler.py",
  "scripts/constant_column_handler.py",
  "scripts/data_cleaner.py",
]

def main() -> None:
  project_root = os.path.dirname(os.path.abspath(__file__))
  os.chdir(project_root)

  for script in SCRIPTS:
    print("\n" + "=" * 60)
    print(f"Running {script}")
    print("=" * 60)
    result = subprocess.run([sys.executable, script], cwd=project_root)
    if result.returncode != 0:
      raise SystemExit(f"{script} failed with exit code {result.returncode}")

  print("\nAll scripts finished successfully.")

if __name__ == "__main__":
  main()
