"""Regression checks for changes that must never skip Docker validation."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from ci_scope import event_paths, jobs_for_paths


class ScopeTests(unittest.TestCase):
    def test_document_and_frontend_boundaries(self):
        self.assertEqual(jobs_for_paths(["README.md", "docs/start.md"]), (False, False))
        self.assertEqual(jobs_for_paths(["README.md", "serving_app/static/index.html"]), (True, False))
        for path in ("serving_app/main.py", "data/features.py", "requirements.txt",
                     "scripts/smoke_test.sh", ".github/workflows/ci.yml", "unknown.file"):
            with self.subTest(path=path):
                self.assertEqual(jobs_for_paths(["README.md", path]), (True, True))

    def test_uncertain_events_require_every_check(self):
        for name, event in (("workflow_dispatch", {}), ("push", {}),
                            ("push", {"before": "0" * 40}),
                            ("pull_request", {"pull_request": {"base": {"sha": "--help"}}})):
            with self.subTest(name=name, event=event):
                self.assertEqual(jobs_for_paths(event_paths(name, event)), (True, True))
        self.assertEqual(jobs_for_paths([]), (True, True))

    def test_entire_push_and_rename_include_backend_change(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL).decode().strip()

            git("init", "-q")
            git("config", "user.name", "CI test")
            git("config", "user.email", "ci@example.invalid")
            (repo / "README.md").write_text("initial")
            (repo / "backend.py").write_text("print('backend')")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            git("mv", "backend.py", "docs.md")
            git("commit", "-qm", "rename backend as documentation")
            (repo / "README.md").write_text("latest docs")
            git("commit", "-qam", "docs-only final commit")
            paths = event_paths("push", {"before": base}, repo)
            self.assertIn("backend.py", paths)
            self.assertEqual(jobs_for_paths(paths), (True, True))
            self.assertEqual(event_paths("push", {"before": "1" * 40}, repo), None)
            last_parent = git("rev-parse", "HEAD^")
            self.assertEqual(jobs_for_paths(event_paths(
                "pull_request", {"pull_request": {"base": {"sha": last_parent}}}, repo
            )), (False, False))


if __name__ == "__main__":
    unittest.main()
