"""Claude Code CLI integration for the QuickAgent persona.

When a user message contains ``!Claude``, QuickAgent delegates the prompt to
the locally-installed ``claude`` CLI (Claude Code) instead of the LiteLLM
model that is configured in Jupyter AI settings.

Public API
----------
has_claude_mention(body)
    Return True if the message body contains a ``!Claude`` / ``!claude`` token.

strip_claude_mention(body)
    Remove the first ``!Claude`` token and return the cleaned text.

find_claude_cli()
    Locate the ``claude`` binary on PATH, returning its path or None.

stream_claude_response(prompt, cwd)
    Async generator that invokes ``claude -p <prompt>`` and yields decoded
    stdout chunks incrementally.  An inline error note is appended if the
    process exits with a non-zero return code.
"""

import asyncio
import os
import re
import shutil
from typing import AsyncGenerator, Optional


# ---------------------------------------------------------------------------
# !Claude trigger helpers
# ---------------------------------------------------------------------------

# Matches !Claude or !claude followed by a word boundary, then optional space.
_CLAUDE_MENTION_RE = re.compile(r"![Cc]laude\b\s*")


def has_claude_mention(body: str) -> bool:
    """Return ``True`` if *body* contains a ``!Claude`` or ``!claude`` token."""
    return bool(_CLAUDE_MENTION_RE.search(body))


def strip_claude_mention(body: str) -> str:
    """Remove the first ``!Claude`` token from *body* and return the result.

    Examples
    --------
    >>> strip_claude_mention("!Claude explain decorators")
    'explain decorators'
    >>> strip_claude_mention("@QuickAgent !Claude fix this bug")
    '@QuickAgent fix this bug'
    """
    return _CLAUDE_MENTION_RE.sub("", body, count=1).strip()


# ---------------------------------------------------------------------------
# CLI discovery
# ---------------------------------------------------------------------------

def find_claude_cli() -> Optional[str]:
    """Return the absolute path of the ``claude`` CLI binary.

    Returns ``None`` when ``claude`` is not found on PATH.
    """
    return shutil.which("claude")


# ---------------------------------------------------------------------------
# Async subprocess invocation
# ---------------------------------------------------------------------------

async def stream_claude_response(
    prompt: str,
    cwd: Optional[str] = None,
    skip_permissions: bool = True,
) -> AsyncGenerator[str, None]:
    """Invoke ``claude -p <prompt>`` and yield decoded stdout chunks.

    The subprocess is invoked with ``asyncio.create_subprocess_exec`` (no
    shell), so special characters in *prompt* are handled safely.

    Stdout is streamed in 512-byte reads so the JupyterLab chat UI updates
    incrementally.  Stderr is captured in a concurrent background task to
    prevent pipe-buffer deadlocks; its content is appended inline if the
    process exits with a non-zero return code.

    Parameters
    ----------
    prompt:
        The text prompt to send to the Claude Code CLI.
    cwd:
        Working directory for the subprocess.  Defaults to the user's home
        directory so that Claude Code can find its ``~/.claude`` config.
    skip_permissions:
        When ``True`` (the default) ``--dangerously-skip-permissions`` is
        passed to the CLI so that Claude Code uses all configured tools
        without interrupting to ask for approval.  Set to ``False`` to
        restore interactive permission prompts.

    Yields
    ------
    str
        Decoded text chunks from the subprocess stdout as they arrive.
        On error, a final chunk containing a formatted error note is yielded
        rather than raising, so the partial response is never lost.

    Raises
    ------
    RuntimeError
        When the ``claude`` binary is not found on PATH (raised *before* the
        first yield, so no partial message is created).
    """
    claude_path = find_claude_cli()
    if claude_path is None:
        raise RuntimeError(
            "Claude Code CLI (`claude`) was not found on PATH.\n\n"
            "Install it with:\n"
            "```bash\n"
            "npm install -g @anthropic-ai/claude-code\n"
            "```\n"
            "See https://claude.ai/code for the full setup guide."
        )

    effective_cwd = cwd or os.path.expanduser("~")

    cmd = [claude_path, "-p", prompt]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=effective_cwd,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    # Drain stderr in a background task to prevent pipe-buffer deadlocks.
    # (stderr output from `claude -p` is typically tiny, but this is robust.)
    stderr_chunks: list[bytes] = []

    async def _drain_stderr() -> None:
        try:
            async for line in proc.stderr:
                stderr_chunks.append(line)
        except asyncio.CancelledError:
            pass

    drain_task = asyncio.create_task(_drain_stderr())

    try:
        # Yield stdout incrementally as it arrives.
        while True:
            chunk = await proc.stdout.read(512)
            if not chunk:
                break
            yield chunk.decode(errors="replace")
    finally:
        # Ensure the subprocess is cleaned up regardless of how the generator exits.
        if proc.returncode is None:
            proc.terminate()
        await proc.wait()
        # Cancel the stderr drain and collect whatever was captured.
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass

    # Surface non-zero exit codes as an inline error note after all content.
    if proc.returncode not in (0, None, -15):  # -15 = SIGTERM (our own terminate)
        stderr_text = b"".join(stderr_chunks).decode(errors="replace").strip()
        error_note = (
            f"\n\n---\n"
            f"⚠️ **Claude CLI exited with code {proc.returncode}**"
        )
        if stderr_text:
            error_note += f"\n```\n{stderr_text}\n```"
        yield error_note
