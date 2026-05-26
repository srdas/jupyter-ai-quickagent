---
fontsize: 12pt
colorlinks: true
linkcolor: blue
geometry:
  - top=1in
  - bottom=1in
  - left=1in
  - right=1in
---

# Beyond Generic AI: Build Custom Agents in Jupyter AI with QuickAgent

by *Sanjiv Das*

Jupyter AI v3.0 brings a paradigm-shifting concept to our favorite data science notebooks: Personas. Here, we present QuickAgent, one persona to rule all agents. 

Instead of interacting with a single, generic chat assistant, Jupyter AI allows you to summon specialized AI collaborators into your sidebar using @ mentions. You can talk to a standard LLM, deploy a specialized software harness like Claude Code, or summon a custom agent tailored to your exact data domain.

If you have spent any time watching autonomous AI agents work, you know the feeling. You give one a task — "analyze this dataset and write me a paper" — and then you just sit back and watch. It reads files, runs code, searches the web, fixes its own errors, and hands you a finished product. No scaffolding. No babysitting. Just results. The first time I saw this work end-to-end, I thought: *this changes everything about how researchers work.*

The trouble is, building an agent from scratch is still, for most people, a software engineering exercise. You wrestle with APIs, tool registrations, prompt templates, and environment variables before you ever get to the interesting part. That friction is what makes implementation difficult for the scientists, educators, and analysts who need it most.

This is the problem that [`jupyter-ai-quickagent`](https://github.com/srdas/jupyter-ai-quickagent) is designed to solve. It is a lightweight Jupyter AI submodule. QuickAgent allows data scientists to build their own agents dynamically through a chat-based interaction.

## What Is QuickAgent?

QuickAgent is a [Jupyter AI](https://jupyter-ai.readthedocs.io/en/v3/) *persona* — a named assistant that lives inside the JupyterLab chat panel. Unlike a generic chatbot, a persona can be extended with custom behavior, tools, and domain knowledge. QuickAgent uses that extension point to let you build, save, and run your own autonomous agents through a simple guided conversation.

QuickAgent was built by looking at the Jupyternaut persona, and extending it with agent-creation capabilities. 

You type `@QuickAgent create` into the Jupyter AI chat, answer five questions, and your agent is ready to run.


## Why is this useful?

[Jupyter AI](https://jupyter-ai.readthedocs.io/en/v3/) already gives you Jupyternaut, a capable assistant that can explain code, generate notebooks, and work with your files. What does an agent layer add? 

Most IDE-native AI assistants force you into a one-size-fits-all model. If you are working on a highly specialized task—such as analyzing climate data or engineering complex financial pipelines—a generic LLM lacks the specialized tools, system prompts, and context required to be genuinely helpful.

While Jupyter AI 3.0 allows for multi-agent workflows, manually coding a new agent persona from scratch every time you switch projects introduces unwanted friction.

The answer is *autonomy* and *specialization*, both provided by QuickAgent. 

- Autonomy: A standard chat assistant waits for you to direct every step. An agent is given a goal — "analyze all the files in this folder and produce a LaTeX research paper," and then it plans, acts, observes the results, and iterates until the goal is achieved. It uses tools: a Python REPL, file I/O, web search, shell commands. It recovers from failures and keeps going until the work is done.

- Specialization: A general-purpose agent is fine for work that is one-off and not intended to be repeated. But if you want an agent that always generates mathematically rigorous exam questions, or one that writes research papers following your specific methodology, you need to inject domain knowledge. QuickAgent does this through *skills* — a folder of Markdown files containing instructions, workflows, and domain context that get woven into the agent's system prompt automatically.

The combination of autonomy plus specialization, configured without code, is what makes this useful. Instead of configuring endless JSON files or writing intricate Python wrappers to define an agent's skills, tools, and behavior, QuickAgent focuses on a chat-based interactive approach to agent creation. You tell Jupyter what you need your persona to do, and QuickAgent helps manifest it as an active, mentionable selection right inside the QuickAgent persona.


## How It Works

QuickAgent is structured to sit seamlessly alongside your Jupyter AI installation. QuickAgent is built on top of [LangChain Deep Agents](https://github.com/langchain-ai/deepagents), a framework for building tool-using agents. It plugs into Jupyter AI through the persona manager, which means it inherits Jupyter AI's model and credential configuration automatically. You set up your API key once in **Settings > Jupyternaut Settings**, and every agent you build uses it. 

Authentication is handled. What remains is defining *what your agent does*.

When you run `@QuickAgent create`, a five-step wizard asks you:

1. What do you want to call this agent?
2. What is its purpose? (A plain-English description — "analyze research datasets and write papers")
3. Which tools should it have access to? (File I/O, Python REPL, shell execution, web fetch, and more)
4. Which search tools? (DuckDuckGo is free and built in; Tavily, Wikipedia, arXiv, and PubMed are optional)
5. Does it have a skills directory? (Point it to a folder of `.md` skill files, and their contents are injected into the system prompt)

The result is saved as a JSON file in `~/.jupyter/jupyter-ai/quickagents/`. This ensures the agent you created is persistent and reusable. You can edit it by hand if you like.

To run it, type: `@QuickAgent use <agent-name>` followed by any prompt. The agent takes it from there.


## QuickAgent makes Data Science easy

Instead of data scientists writing all the code, their role is evolving into that of a principal investigator orchestrating an elite team of specialized AI agents. You might have one agent that specializes in fetching remote geospatial data formats, another that excels at cleaning data frames, and a third designed to optimize machine learning hyperparameters.

The traditional boundaries between writing code and instructing an AI are blurring. QuickAgent highlights a critical trend: agentic hierarchies. By lowering the barrier to entry for creating these personas, repositories like `jupyter-ai-quickagent` are making specialized AI accessible to everyday developers—no complex backend engineering required.


## Two Examples

### The MathQ Agent

A colleague wrote an [essay](https://github.com/srdas/skill-collection/blob/main/math_questions/Good%20Math%20Questions.pdf) on what makes a good mathematics question — the kind that tests deep conceptual understanding rather than mechanical computation. We converted this essay into a [`SKILL.md`](https://github.com/srdas/skill-collection/blob/main/math_questions/skills/SKILL.md) file and pointed a QuickAgent at it.

The result is an agent that generates exam questions aligned to those criteria. It does not just produce any question — it produces questions it can defend as *good* questions, explaining why each one tests reasoning rather than recall. Whether it always gets this right is a matter of judgment. But the framework — injecting expert human criteria into an agent's decision-making via a skills file — is a useful pattern. You can find the MathQ agent and its associated skills file [here](https://github.com/srdas/jupyter-ai-quickagent#quick-start). 

![](static/mathq_question.png)

![](static/mathq_answer.png)

![](static/mathq_why.png)

### The Researcher Agent

You point the Researcher agent at a folder containing a CSV dataset and any other files (paper, dscriptions of the data, etc.). The agent reads the data, formulates research questions, writes and executes Python code to produce statistical analyses and visualizations, generates a reproducible Jupyter notebook, and produces a complete LaTeX paper. This agent needs a frontier LLM, and Claude Opus-4.7 is recommended. 

The output is first-pass research work, not just a summary, but tables, figures, statistical tests, and a structured paper you could work from as a first draft. The [skills directory](https://github.com/srdas/skill-collection/tree/main/researcher/skills) for this agent contains a set of `.md` files defining what good empirical research looks like, what a rigorous paper structure entails, and how to handle common data quality issues. The agent consults all of it and prepares the outputs, leveraging Jupyter notebooks to provide fully reproducible results. 

We are at a moment where AI can genuinely accelerate scientific work, not by replacing judgment, but by handling the execution layer. The researcher does not even need to define the question, or design the methodology, evaluates the output, but must take responsibility for the conclusions. But the hours of boilerplate coding, formatting, and mechanical analysis? Those can be delegated to the agent and verification and checking becomes an important human exercise. Of course, the human can influence all parts of this process. 

The bottleneck has always been the gap between "I have a dataset" and "I have a result." QuickAgent narrows that gap. A well-designed agent, given good skills and clear direction, can turn a dataset into a draft paper in the time it used to take to just set up the analysis environment. The Researcher example in this repository does exactly that, with documented output you can inspect yourself. You can also see additional examples of generated research in the [`examples`](https://github.com/srdas/jupyter-ai-quickagent/tree/main/examples) folder. 

The deeper point is this: the most valuable thing about configuring agents with *tools* and *skills* is that it encodes expertise. An experienced researcher's intuitions about what constitutes a good analysis, what a rigorous paper looks like, what pitfalls to avoid — all of that can be written down in Markdown and given to an agent. The agent then applies it consistently, at scale, across every task. You are not just automating work. 

It is not only about research. Any agent can be spun up quickly. Here we only highlighted two agents, one that generates good math questions, and the other that does autonomous research. 


## An invitation to Jupyter AI

It is worth being explicit about the layering here. [Jupyter AI](https://jupyter-ai.readthedocs.io/en/v3/) is the foundation, an extensible AI integration for JupyterLab that handles model configuration, the chat UI, magic commands, and the persona manager. QuickAgent is an add-on to that foundation, implementing one specific persona focused on autonomous agent creation and execution.

If you are not already using Jupyter AI, the QuickAgent installer sets everything up for you. If you are, QuickAgent slots in alongside Jupyternaut and any other personas you have installed. The two complement each other well — Jupyternaut for conversational code assistance, QuickAgent for autonomous multi-step tasks.

The full Jupyter AI documentation is at [jupyter-ai.readthedocs.io/en/v3/](https://jupyter-ai.readthedocs.io/en/v3/) and is worth reading to understand the broader ecosystem.


## Get Started

The repository is at [github.com/srdas/jupyter-ai-quickagent](https://github.com/srdas/jupyter-ai-quickagent). Clone it, follow the [installation instructions](https://github.com/srdas/jupyter-ai-quickagent/tree/main#quick-installation-with-jupyter-ai) in the README, and you can get up and running. 

### Option 1: Full JupyterLab Install (Recommended)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) first — it is the fastest Python package manager available and makes the rest of this trivial.

Then download the [installer script](https://github.com/srdas/jupyter-ai-quickagent/blob/main/install_nondev.sh) and run it:

```bash
chmod a+x install_nondev.sh
./install_nondev.sh
```

That single script installs Jupyter AI, all its dependencies, and QuickAgent, then launches JupyterLab. For subsequent sessions:

```bash
cd jupyter-ai-quickagents
source .venv/bin/activate
jupyter lab
```

Once JupyterLab is open, configure your model in **Settings > Jupyternaut Settings** (remember it is based on Jupyternaut). Any provider supported by [LiteLLM](https://docs.litellm.ai/) works — OpenAI, Anthropic, Google, Azure, AWS Bedrock, and dozens more. Then open the chat panel, and start building quick agents.

### Option 2: CLI-Only Install

If you want to run agents from the terminal without JupyterLab, the CLI install is just as straightforward:

```bash
git clone https://github.com/srdas/jupyter-ai-quickagent.git
cd jupyter-ai-quickagent
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r CLI-requirements.txt
```

Connect it to a model:

```bash
export QUICKAGENT_MODEL='us.anthropic.claude-opus-4-6-v1'
```

Create your first agent:

```bash
python -m jupyter_ai_quickagent create
```

Run it:

```bash
python -m jupyter_ai_quickagent run MyAgent "Summarize the key findings in this folder"
```

### Your First Agent in Three Commands (JupyterLab)

Once you are inside the JupyterLab chat panel:

```
@QuickAgent create
```

Follow the five prompts. Then activate your new agent:

```
@QuickAgent use MyAgent
```

And put it to work:

```
Analyze the data in ~/Downloads/survey_results and write a summary report
```

That is it. The agent plans its approach, reads the files, runs the analysis, and writes the report — all without further input from you.

This is just "One persona to rule all agents."
