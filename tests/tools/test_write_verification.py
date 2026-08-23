"""Tests for write_file post-write content verification (verified flag)."""

import json
from unittest.mock import patch as mock_patch

import pytest

from tools.file_tools import write_file_tool


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path


class TestWriteVerification:
    def test_successful_write_reports_verified(self, workdir):
        f = workdir / "out.txt"
        r = json.loads(write_file_tool(str(f), "hello verified world\n", task_id="t-wv"))
        assert r.get("bytes_written") == len("hello verified world\n")
        assert r.get("verified") is True

    def test_unicode_content_verified(self, workdir):
        f = workdir / "uni.txt"
        content = "línea → uno · ✓\n"
        r = json.loads(write_file_tool(str(f), content, task_id="t-wv"))
        assert r.get("verified") is True

    def test_crlf_preservation_still_verifies(self, workdir):
        # Existing CRLF file: write_file converts LF content to CRLF before
        # writing; verification hashes the shim-adjusted content, so it must
        # still report verified.
        f = workdir / "win.txt"
        f.write_bytes(b"old line\r\n")
        r = json.loads(write_file_tool(str(f), "new line\nsecond\n", task_id="t-wv"))
        assert "error" not in r
        assert r.get("verified") is True
        assert b"\r\n" in f.read_bytes()

    def test_hash_mismatch_is_hard_error(self, workdir):
        f = workdir / "bad.txt"
        import tools.file_operations as fo

        real_write = fo.ShellFileOperations._atomic_write

        def write_then_corrupt(self, path, content):
            result = real_write(self, path, content)
            from pathlib import Path as _P
            _P(path).write_text("CORRUPTED\n")
            return result

        with mock_patch.object(fo.ShellFileOperations, "_atomic_write", write_then_corrupt):
            r = json.loads(write_file_tool(str(f), "actual content\n", task_id="t-wv"))
        assert "error" in r
        assert "did not persist" in r["error"]
        assert r.get("bytes_written") == len("actual content\n")

    def test_verification_failure_never_breaks_write(self, workdir):
        # sha256sum unavailable/failing -> native host-path fallback still
        # verifies on LocalEnvironment, write still ok.
        f = workdir / "ok.txt"
        import tools.file_operations as fo

        real_exec = fo.ShellFileOperations._exec

        def flaky_exec(self, cmd, **kw):
            if "sha256sum" in cmd or "shasum" in cmd:
                raise RuntimeError("no hash binary")
            return real_exec(self, cmd, **kw)

        with mock_patch.object(fo.ShellFileOperations, "_exec", flaky_exec):
            r = json.loads(write_file_tool(str(f), "content lands anyway\n", task_id="t-wv2"))
        assert "error" not in r
        assert f.read_text() == "content lands anyway\n"
        assert r.get("verified") is True

    def test_polluted_sha256sum_stdout_still_verifies(self, workdir):
        f = workdir / "msl.txt"
        import tools.file_operations as fo
        from tools.file_operations import ExecuteResult

        real_exec = fo.ShellFileOperations._exec

        def msl_exec(self, cmd, **kw):
            r = real_exec(self, cmd, **kw)
            if "sha256sum" in cmd or "shasum" in cmd:
                noise = (
                    "bash(12345) MallocStackLogging: recording malloc "
                    "(and VM allocation) stacks using lite mode\n"
                )
                return ExecuteResult(stdout=noise + r.stdout, exit_code=r.exit_code)
            return r

        with mock_patch.object(fo.ShellFileOperations, "_exec", msl_exec):
            r = json.loads(write_file_tool(str(f), "hello msl\n", task_id="t-wv-msl"))
        assert "error" not in r
        assert r.get("verified") is True
        assert f.read_text() == "hello msl\n"

    def test_wrong_shell_hash_recovers_via_native_read(self, workdir):
        f = workdir / "native.txt"
        import tools.file_operations as fo
        from tools.file_operations import ExecuteResult

        real_exec = fo.ShellFileOperations._exec

        def wrong_hash_exec(self, cmd, **kw):
            r = real_exec(self, cmd, **kw)
            if "sha256sum" in cmd or "shasum" in cmd:
                return ExecuteResult(stdout=("0" * 64) + "  " + str(f), exit_code=0)
            return r

        with mock_patch.object(fo.ShellFileOperations, "_exec", wrong_hash_exec):
            r = json.loads(write_file_tool(str(f), "native recovers\n", task_id="t-wv-nat"))
        assert "error" not in r
        assert r.get("verified") is True
        assert f.read_text() == "native recovers\n"

    def test_polluted_size_probe_still_reads(self, workdir):
        from tools.file_tools import read_file_tool
        f = workdir / "read-msl.txt"
        f.write_text("line one\nline two\n")
        import tools.file_operations as fo
        from tools.file_operations import ExecuteResult

        real_exec = fo.ShellFileOperations._exec

        def msl_exec(self, cmd, **kw):
            r = real_exec(self, cmd, **kw)
            if "wc -c" in cmd or "[ -f " in cmd:
                noise = (
                    "bash(1) MallocStackLogging: recording malloc "
                    "(and VM allocation) stacks using lite mode\n"
                )
                return ExecuteResult(stdout=noise + r.stdout, exit_code=r.exit_code)
            return r

        with mock_patch.object(fo.ShellFileOperations, "_exec", msl_exec):
            r = json.loads(read_file_tool(str(f), task_id="t-wv-read"))
        assert "error" not in r
        assert r.get("file_size") == len("line one\nline two\n")
        assert "line one" in r.get("content", "")
        assert "File is empty" not in (r.get("hint") or "")

    def test_polluted_verify_cat_still_patches(self, workdir):
        from tools.file_tools import patch_tool
        f = workdir / "patch-msl.txt"
        f.write_text("hello world\n")
        import tools.file_operations as fo
        from tools.file_operations import ExecuteResult

        real_exec = fo.ShellFileOperations._exec
        cat_calls = {"n": 0}

        def msl_exec(self, cmd, **kw):
            r = real_exec(self, cmd, **kw)
            # Pollute only the post-write verify cat (2nd `cat 'path'`), not
            # the initial read — baking MSL into the write is a different bug
            # that the capture-boundary strip prevents in production.
            if cmd.strip().startswith("cat "):
                cat_calls["n"] += 1
                if cat_calls["n"] >= 2:
                    noise = (
                        "bash(1) MallocStackLogging: recording malloc "
                        "(and VM allocation) stacks using lite mode\n"
                    )
                    return ExecuteResult(stdout=r.stdout + noise, exit_code=r.exit_code)
            return r

        with mock_patch.object(fo.ShellFileOperations, "_exec", msl_exec):
            r = json.loads(patch_tool(
                path=str(f), old_string="world", new_string="hermes",
                task_id="t-wv-patch",
            ))
        assert r.get("success") is True
        assert "error" not in r
        assert f.read_text() == "hello hermes\n"
        assert "MallocStackLogging" not in f.read_text()
