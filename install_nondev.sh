git clone https://github.com/srdas/jupyter-ai-quickagent.git

cd jupyter-ai-quickagent

uv venv --python 3.12
sleep 5
source .venv/bin/activate

uv pip install jupyter-ai

npm install -g @agentclientprotocol/claude-agent-acp

uv pip install jupyter-ai-magic-commands

uv pip install jupyter-ai-jupyternaut

cd ..
uv pip install -e jupyter-ai-quickagent

cd jupyter-ai-quickagent 
jupyter lab
