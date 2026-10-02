"""Select Docker checks from the complete event diff; uncertainty runs both."""
import json
import os
import re
import subprocess
from pathlib import Path

DOC_FILES = {"README.md", "RULES.md", "AGENTS.md", "LICENSE", "LICENSE.txt"}
SHA = re.compile(r"[0-9a-fA-F]{40}")


def jobs_for_paths(paths):
    if not paths:
        return True, True
    runtime = False
    for path in paths:
        if path in DOC_FILES or (path.startswith("docs/") and path.endswith(".md")):
            continue
        if path.startswith("serving_app/static/"):
            runtime = True
            continue
        return True, True
    return runtime, False


def event_paths(event_name, event, repo="."):
    if event_name == "pull_request":
        base = event.get("pull_request", {}).get("base", {}).get("sha", "")
    elif event_name == "push":
        base = event.get("before", "")
    else:
        return None
    if not isinstance(base, str) or not SHA.fullmatch(base) or base == "0" * 40:
        return None
    try:
        subprocess.run(
            ["git", "cat-file", "-e", base + "^{commit}"], cwd=repo,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-only", "-z", base, "HEAD", "--"],
            cwd=repo,
        )
    except subprocess.CalledProcessError:
        return None
    return [os.fsdecode(path) for path in diff.split(b"\0") if path]


def main():
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        paths = event_paths(os.environ.get("GITHUB_EVENT_NAME", ""), event)
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        paths = None
    runtime, trained = jobs_for_paths(paths)
    output = f"runtime={str(runtime).lower()}\ntrained={str(trained).lower()}\n"
    print(output, end="")
    if paths is None:
        print("Diff unavailable or manual run: full Docker validation.")
    else:
        print(f"Changed files: {len(paths)}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(output)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(
                f"### Selected checks\n\nStatic checks: always. "
                f"Docker runtime: {runtime}. Docker trained: {trained}.\n"
            )


if __name__ == "__main__":
    main()
