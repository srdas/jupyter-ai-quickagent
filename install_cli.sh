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
  curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash
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
