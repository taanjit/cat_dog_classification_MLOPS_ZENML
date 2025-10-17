# steps/data_version_checker.py
import subprocess
import os
from zenml import step


@step(enable_cache=False)
def data_version_checker(data_dir: str) -> bool:
    """
    Step: Automatically updates and checks DVC dataset version.

    - Runs `dvc add` for train/validation data.
    - Commits and pushes to remote if changes exist.
    - Detects if data has changed based on DVC status.
    Returns:
        True  -> retrain required (dataset updated)
        False -> no retraining needed
    """
    print("[data_version_checker] 🔍 Checking and syncing DVC data...")

    # Ensure data directory exists
    if not os.path.exists(data_dir):
        print(f"[data_version_checker] ❌ Data directory not found: {data_dir}")
        return False

    try:
        # --- 1️⃣ DVC Add: track dataset changes ---
        print("[data_version_checker] ➕ Running 'dvc add' for data folders...")
        subprocess.run(["dvc", "add", "data/train", "data/validation"], check=False)

        # --- 2️⃣ DVC Commit: record updated hash locally ---
        print("[data_version_checker] 📝 Committing DVC changes...")
        subprocess.run(["dvc", "commit"], check=False)

        # --- 3️⃣ DVC Status: detect uncommitted changes ---
        print("[data_version_checker] 🔎 Checking DVC status...")
        result = subprocess.run(
            ["dvc", "status", "-q"],
            capture_output=True,
            text=True,
            check=False
        )

        has_changes = result.stdout.strip() != ""
        if has_changes:
            print("[data_version_checker] ⚠️ Dataset changes detected. Syncing to remote...")

            # --- 4️⃣ DVC Push: sync to remote if configured ---
            push_result = subprocess.run(
                ["dvc", "push"],
                capture_output=True,
                text=True,
                check=False
            )

            if push_result.returncode == 0:
                print("[data_version_checker] ☁️ Data successfully pushed to remote storage.")
            else:
                print(f"[data_version_checker] ⚠️ DVC push failed or no remote configured: {push_result.stderr}")

            print("[data_version_checker] ✅ Dataset updated — retraining required.")
            return True
        else:
            print("[data_version_checker] ✅ No dataset changes detected — skipping retraining.")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[data_version_checker] ⚠️ DVC command failed: {e}")
        return True  # retrain just in case

    except Exception as e:
        print(f"[data_version_checker] ⚠️ Unexpected error: {e}")
        return True
