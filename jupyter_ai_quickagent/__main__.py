"""CLI entry point for jupyter_ai_quickagent.

Run saved QuickAgents from the command line without JupyterLab.

Usage:
    quickagent list
    quickagent info <name>
    quickagent create
    quickagent delete <name>
    quickagent run <name> <message>
    quickagent run <name>              (interactive / stdin)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .agent_store import (
    COMMON_TOOLS,
    SEARCH_TOOLS,
    AgentConfig,
    delete_agent,
    list_agents,
    load_agent,
    save_agent,
)
from .prompt_template import QUICKAGENT_SYSTEM_PROMPT_TEMPLATE


def _get_model(model_id: str):
    """Build a ChatLiteLLM instance from a model string like 'anthropic/claude-sonnet-4-20250514'."""
    from langchain_community.chat_models import ChatLiteLLM

    return ChatLiteLLM(model=model_id, streaming=True)


def _resolve_model(args_model: str | None) -> str:
    """Resolve the model ID from CLI flag or environment variable."""
    model = args_model or os.environ.get("QUICKAGENT_MODEL")
    if not model:
        print(
            "Error: No model specified.\n"
            "Use --model MODEL or set the QUICKAGENT_MODEL environment variable.\n"
            "Examples:\n"
            "  quickagent run MyAgent 'hello' --model anthropic/claude-sonnet-4-20250514\n"
            "  export QUICKAGENT_MODEL=openai/gpt-4o",
            file=sys.stderr,
        )
        sys.exit(1)
    return model


def _load_skill_content(skills_dir: str) -> str:
    """Load all .md files from a skills directory."""
    if not skills_dir or not os.path.isdir(skills_dir):
        return ""

    sections = []
    for dirpath, _, filenames in os.walk(skills_dir):
        for filename in sorted(filenames):
            if filename.endswith(".md"):
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, skills_dir)
                try:
                    with open(filepath) as f:
                        content = f.read()
                    sections.append(f"--- Skill: {rel_path} ---\n{content}")
                except OSError:
                    pass
    return "\n\n".join(sections)


def _find_project_root(skills_dir: str) -> str:
    """Walk up from the skills directory to find the project root."""
    path = os.path.abspath(skills_dir)
    parts = path.split(os.sep)
    if ".claude" in parts:
        idx = parts.index(".claude")
        return os.sep.join(parts[:idx])
    current = path
    for _ in range(10):
        parent = os.path.dirname(current)
        if parent == current:
            break
        if any(
            os.path.exists(os.path.join(parent, marker))
            for marker in (".git", ".claude", "pyproject.toml", "package.json")
        ):
            return parent
        current = parent
    return os.path.dirname(skills_dir)


def cmd_list(args):
    agents = list_agents()
    if not agents:
        print("No saved agents. Use 'quickagent create' to make one.")
        return
    print("Saved agents:\n")
    for a in agents:
        tools = ", ".join(a.tools) if a.tools else "none"
        search = ", ".join(a.search_tools) if a.search_tools else "none"
        skills = a.skills_dir if a.skills_dir else "none"
        print(f"  {a.name}")
        print(f"    Purpose: {a.purpose}")
        print(f"    Tools:   {tools}")
        print(f"    Search:  {search}")
        print(f"    Skills:  {skills}")
        print()


def cmd_info(args):
    config = load_agent(args.name)
    if not config:
        print(f"Agent '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)
    tools = ", ".join(config.tools) if config.tools else "none"
    search = ", ".join(config.search_tools) if config.search_tools else "none"
    skills = config.skills_dir if config.skills_dir else "none"
    print(f"Name:       {config.name}")
    print(f"Purpose:    {config.purpose}")
    print(f"Tools:      {tools}")
    print(f"Search:     {search}")
    print(f"Skills dir: {skills}")


def cmd_delete(args):
    if delete_agent(args.name):
        print(f"Deleted agent '{args.name}'.")
    else:
        print(f"Agent '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)


def cmd_create(args):
    print("=== Create a new QuickAgent ===\n")

    name = input("Agent name: ").strip()
    if not name:
        print("Aborted.", file=sys.stderr)
        sys.exit(1)

    purpose = input("Purpose (what should it do): ").strip()
    if not purpose:
        print("Aborted.", file=sys.stderr)
        sys.exit(1)

    print("\nAvailable tools:")
    for i, (t, desc) in enumerate(COMMON_TOOLS.items(), 1):
        print(f"  {i}. {t} — {desc}")
    print("\nEnter tool names (comma-separated), 'all', 'default', or 'none':")
    tools_input = input("Tools: ").strip()
    tools = _parse_tools(tools_input, COMMON_TOOLS)

    print("\nAvailable search tools:")
    for i, (t, desc) in enumerate(SEARCH_TOOLS.items(), 1):
        print(f"  {i}. {t} — {desc}")
    print("\nEnter search tool names (comma-separated), 'all', or 'none':")
    search_input = input("Search tools: ").strip()
    search_tools = _parse_tools(search_input, SEARCH_TOOLS)

    print("\nSkills directory (path to folder with .md files, or 'none'):")
    skills_input = input("Skills dir: ").strip()
    skills_dir = ""
    if skills_input and skills_input.lower() not in ("none", "skip", ""):
        skills_dir = os.path.abspath(os.path.expanduser(skills_input))

    config = AgentConfig(
        name=name,
        purpose=purpose,
        tools=tools,
        search_tools=search_tools,
        skills_dir=skills_dir,
    )

    print(f"\n  Name:    {config.name}")
    print(f"  Purpose: {config.purpose}")
    print(f"  Tools:   {', '.join(config.tools) or 'none'}")
    print(f"  Search:  {', '.join(config.search_tools) or 'none'}")
    print(f"  Skills:  {config.skills_dir or 'none'}")

    confirm = input("\nSave this agent? [Y/n] ").strip().lower()
    if confirm in ("", "y", "yes"):
        path = save_agent(config)
        print(f"\nSaved to {path}")
    else:
        print("Cancelled.")


def _parse_tools(text: str, available: dict[str, str]) -> list[str]:
    lower = text.strip().lower()
    if lower == "none" or lower == "":
        return []
    if lower == "all":
        return list(available.keys())
    if lower == "default" and available is COMMON_TOOLS:
        return ["execute", "read_file", "write_file", "edit_file", "ls", "grep"]
    selected = []
    for part in text.split(","):
        name = part.strip().lower()
        if name in available:
            selected.append(name)
    return selected


def _build_agent(config: AgentConfig, model):
    """Build a deep agent from config."""
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend

    tools = _resolve_tools(config)

    backend = LocalShellBackend(
        root_dir=Path.home(),
        virtual_mode=False,
    )

    skills_dir = config.skills_dir
    skills_dir_parent = _find_project_root(skills_dir) if skills_dir else ""
    skill_content = _load_skill_content(skills_dir)

    system_prompt = QUICKAGENT_SYSTEM_PROMPT_TEMPLATE.render(
        persona_name=config.name,
        purpose=config.purpose,
        context="",
        skills=skill_content,
        skills_dir=skills_dir,
        skills_dir_parent=skills_dir_parent,
        home_dir=os.path.expanduser("~"),
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
    )


def _make_web_fetch_tool():
    from langchain_core.tools import tool

    @tool
    def web_fetch(url: str) -> str:
        """Fetch and return the text content of a web page at the given URL."""
        import httpx
        from html.parser import HTMLParser

        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._pieces: list[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "noscript"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "noscript"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    text = data.strip()
                    if text:
                        self._pieces.append(text)

            def get_text(self) -> str:
                return "\n".join(self._pieces)

        try:
            resp = httpx.get(url, follow_redirects=True, timeout=30)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"Error fetching URL: {e}"

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            extractor = _TextExtractor()
            extractor.feed(resp.text)
            return extractor.get_text()[:50000]
        else:
            return resp.text[:50000]

    return web_fetch


def _get_search_tool(name: str):
    try:
        if name == "tavily_search":
            from langchain_community.tools.tavily_search import TavilySearchResults
            return TavilySearchResults(max_results=5)
        elif name == "duckduckgo_search":
            from langchain_community.tools import DuckDuckGoSearchResults
            return DuckDuckGoSearchResults()
        elif name == "wikipedia":
            from langchain_community.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper
            return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        elif name == "arxiv":
            from langchain_community.tools import ArxivQueryRun
            from langchain_community.utilities import ArxivAPIWrapper
            return ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        elif name == "pubmed":
            from langchain_community.tools import PubmedQueryRun
            from langchain_community.utilities import PubMedAPIWrapper
            return PubmedQueryRun(api_wrapper=PubMedAPIWrapper())
    except (ImportError, Exception):
        pass
    return None


def _resolve_tools(config: AgentConfig) -> list:
    tools = []
    for tool_name in config.tools:
        if tool_name == "web_fetch":
            tools.append(_make_web_fetch_tool())
    for tool_name in config.search_tools:
        tool = _get_search_tool(tool_name)
        if tool:
            tools.append(tool)
    return tools


async def _run_agent_async(agent, message: str):
    """Run the agent and print streamed output."""
    async for token, metadata in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")
        if node == "model" and hasattr(token, "text") and token.text:
            print(token.text, end="", flush=True)
    print()


def cmd_run(args):
    model_id = _resolve_model(args.model)
    config = load_agent(args.name)
    if not config:
        print(f"Agent '{args.name}' not found. Use 'quickagent list' to see agents.", file=sys.stderr)
        sys.exit(1)

    message = " ".join(args.message) if args.message else None

    if not message:
        if sys.stdin.isatty():
            print(f"Running agent '{config.name}'. Type your message (Ctrl+D to send):")
            message = sys.stdin.read().strip()
        else:
            message = sys.stdin.read().strip()

    if not message:
        print("No message provided.", file=sys.stderr)
        sys.exit(1)

    model = _get_model(model_id)
    agent = _build_agent(config, model)
    asyncio.run(_run_agent_async(agent, message))


def main():
    parser = argparse.ArgumentParser(
        prog="quickagent",
        description="Run Jupyter AI QuickAgents from the command line.",
    )
    parser.add_argument(
        "--model", "-m",
        help="LiteLLM model ID (e.g. anthropic/claude-sonnet-4-20250514, openai/gpt-4o). "
             "Can also be set via QUICKAGENT_MODEL env var.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List all saved agents")

    p_info = subparsers.add_parser("info", help="Show agent details")
    p_info.add_argument("name", help="Agent name")

    subparsers.add_parser("create", help="Interactively create a new agent")

    p_delete = subparsers.add_parser("delete", help="Delete a saved agent")
    p_delete.add_argument("name", help="Agent name")

    p_run = subparsers.add_parser("run", help="Run a saved agent with a message")
    p_run.add_argument("name", help="Agent name")
    p_run.add_argument("message", nargs="*", help="Message to send (or pipe via stdin)")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
