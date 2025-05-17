---
title: GAIA Agent
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---
### Final Agent HF Course

This project is part of the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction). For more information about the course, syllabus, and certification process, visit the [course introduction page](https://huggingface.co/learn/agents-course/unit0/introduction).

You can find and try the agent in my Hugging Face Space here: [serverdaun/final_gaia_agent_hf_course](https://huggingface.co/spaces/serverdaun/final_gaia_agent_hf_course).

---

## GAIA Benchmark Target

This agent is designed to participate in the [GAIA benchmark for General AI Assistants](https://huggingface.co/gaia-benchmark). GAIA is a comprehensive benchmark for evaluating the capabilities of general AI agents across a wide range of tasks. The benchmark is maintained by the Hugging Face community and features a public [leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard) for submissions and results.

For more information about GAIA, its datasets, and the leaderboard, visit the [GAIA organization page](https://huggingface.co/gaia-benchmark).


## Agent Logic Overview

### Architecture
This project implements a modular agent using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain) frameworks. The agent is orchestrated as a state graph, where each node represents a step in the reasoning or tool-use process. The core LLM is accessed via Azure OpenAI, and the agent is designed to invoke a variety of tools to solve complex tasks.

### Tools
The agent is equipped with a rich set of tools, including:
- **Search Tools**: Wikipedia, Tavily, and Arxiv search for retrieving information from the web and scientific literature.
- **Math Tools**: Arithmetic operations, power, square root, modulus, and group theory utilities (commutativity, associativity, identity, inverses).
- **Web Scraping**: Extracts main content from arbitrary web pages.
- **Image Analysis**: Uses Azure OpenAI's vision capabilities to answer questions about images.
- **Audio Transcription**: Transcribes audio files using Whisper.
- **Code Execution**: Runs code files in various languages (Python, JS, TS, Bash, Ruby, PHP, Go) and returns output/errors.
- **Tabular Data Tools**: Summarizes, filters, and manipulates CSV, Excel, and Parquet files.

### Agent Workflow
1. **Initialization**: The agent is built using a state graph, with nodes for the LLM and tool invocation. The LLM is bound to the available tools.
2. **Receiving Questions**: The Gradio app fetches a set of questions (some with associated files) from a remote API.
3. **Processing**: For each question, the agent constructs a message history (including a system prompt and the user question/file path) and invokes the LLM. If the LLM decides a tool is needed, the appropriate tool is called and the result is fed back into the conversation.
4. **Answer Extraction**: The agent's final answer is parsed and submitted back to the evaluation server.
5. **Submission**: All answers are submitted in batch, and the results (including score and feedback) are displayed in the Gradio interface.

### Extending the Agent
- **Adding Tools**: Implement a new function in `tools.py` and decorate it with `@tool`. Add it to the `TOOLS` list in `agent.py`.
- **Modifying Logic**: Adjust the state graph in `agent.py` or the agent invocation logic in `app.py` as needed.