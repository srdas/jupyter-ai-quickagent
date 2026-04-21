#!/usr/bin/env bash
#
# install.sh — Clone and install jupyter-ai-devrepo with all submodules.
#
# Prompts for a GitHub Password once and then re-uses it 
# for cloning the main repo and all submodules via HTTPS.
#
# Prerequisites: git, uv, just
#
# Usage:
#   chmod +x install.sh
#   ./install.sh [target-directory]
#
# If target-directory is omitted, clones into ./jupyter-ai-devrepo

# USAGE
# From wherever you want the devrepo to live:                                                                                      
#
# ./install.sh              # clones into ./jupyter-ai-devrepo                                                                      # 
# ./install.sh my-folder    # clones into ./my-folder   


set -euo pipefail

REPO_ORG="jupyter-ai-contrib"
REPO_NAME="jupyter-ai-devrepo"
TARGET_DIR="${1:-$REPO_NAME}"

# ── Preflight checks ────────────────────────────────────────────────
check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "Error: '$1' is not installed. Please install it first." >&2
    echo "  On macOS:  brew install $1" >&2
    exit 1
  fi
}

check_cmd git
check_cmd uv
check_cmd just

# ── Prompt for GitHub credentials (once) ─────────────────────────────
echo "This script clones via HTTPS so you only need to authenticate once."
echo ""
read -rp "GitHub username: " GH_USER
read -rsp "GitHub password: " GH_PASS
echo ""

if [[ -z "$GH_USER" || -z "$GH_PASS" ]]; then
  echo "Error: username and password must not be empty." >&2
  exit 1
fi

# Build the authenticated HTTPS base URL.
AUTH_URL="https://${GH_USER}:${GH_PASS}@github.com"

# Tell git to rewrite all github.com SSH and HTTPS URLs to use our
# authenticated URL.  This is scoped to this script's process and any
# child processes — it does NOT modify the user's global git config.
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0="url.${AUTH_URL}/.insteadOf"
export GIT_CONFIG_VALUE_0="git@github.com:"
export GIT_CONFIG_KEY_1="url.${AUTH_URL}/.insteadOf"
export GIT_CONFIG_VALUE_1="https://github.com/"

# ── Clone ────────────────────────────────────────────────────────────
echo ""
echo "Cloning ${REPO_ORG}/${REPO_NAME} into ${TARGET_DIR}/ ..."
git clone --recurse-submodules \
  "${AUTH_URL}/${REPO_ORG}/${REPO_NAME}.git" \
  "$TARGET_DIR"

cd "$TARGET_DIR"

# ── Collect all with optional submodules──────────────────────────────
just pull-all

# ── Pull latest changes on all submodules ────────────────────────────
echo ""
echo "Pulling latest changes on all submodules ..."
just mainline all

# ── Optional: clone jupyter-ai-quickagent ────────────────────────────
echo ""
read -rp "Clone and install jupyter-ai-quickagent? [y/N] " qa_answer
if [[ "$qa_answer" =~ ^[Yy]$ ]]; then
  echo "Cloning jupyter-ai-quickagent ..."
  git clone https://github.com/srdas/jupyter-ai-quickagent.git
fi

# ── Install with optional submodules──────────────────────────────
just sync-all --refresh

# ── Install all packages ─────────────────────────────────────────────
echo ""
echo "Installing all packages (this may take a while) ..."
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$PWD/.venv/bin:$PATH"
just install-all

# ── AI Persona Installation ──────────────────────────────────────────
echo ""
echo "============================================"
echo "  Optional AI Persona Installation"
echo "============================================"
echo ""

ask_install() {
  local name="$1"
  read -rp "Install ${name}? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]]
}

# @Claude
if ask_install "@Claude (claude-agent-acp via npm)"; then
  echo "Installing claude-agent-acp ..."
  npm install -g @agentclientprotocol/claude-agent-acp
  read -rp "  Set CLAUDE_CODE_EXECUTABLE path? (leave blank to use default): " claude_exec
  if [[ -n "$claude_exec" ]]; then
    export CLAUDE_CODE_EXECUTABLE="$claude_exec"
    echo "  CLAUDE_CODE_EXECUTABLE=${claude_exec}"
  fi
fi

# @Gemini
if ask_install "@Gemini (gemini-cli >= 0.34.0 via npm)"; then
  echo "Installing gemini-cli ..."
  npm install -g @google/gemini-cli
fi

# @Kiro
if ask_install "@Kiro (kiro-cli >= 1.25.0 via kiro.dev)"; then
  echo "Installing kiro-cli ..."
  curl -fsSL https://cli.kiro.dev/install | bash
fi

# @Mistral-Vibe
if ask_install "@Mistral-Vibe (vibe-acp via uv)"; then
  echo "Installing mistral-vibe ..."
  uv tool install mistral-vibe
fi

# @OpenCode
if ask_install "@OpenCode (opencode-ai >= 1.0.0 via npm)"; then
  echo "Installing opencode-ai ..."
  npm install -g opencode-ai
fi

# @Codex
if ask_install "@Codex (codex-acp via npm)"; then
  echo "Installing codex-acp ..."
  npm install -g @zed-industries/codex-acp
fi

# @Goose
if ask_install "@Goose (goose-cli >= 1.8.0 via block/goose)"; then
  echo "Installing goose ..."
  curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
fi

# ── Optional boto3 Installation ──────────────────────────────────────
if ask_install "boto3 (AWS SDK for Python)"; then
  echo "Installing boto3 ..."
  uv pip install boto3
fi

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  To start JupyterLab:"
echo "    cd ${TARGET_DIR}"
echo "    source .venv/bin/activate"
echo "    just start"
echo "============================================"
