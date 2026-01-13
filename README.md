🧠 Autonomous Data Science Team (Local, Agentic AI)

A fully autonomous, multi-agent data science system built using Agno, Ollama, and SQLite, designed to run entirely on your local machine.

This project orchestrates a team of specialized AI agents to handle end-to-end data science workflows—from raw CSV ingestion to model training, evaluation, and debugging.

🚀 Key Highlights

🧑‍🤝‍🧑 Multi-Agent Architecture (Team-based, not single LLM)

🧠 Autonomous Task Decomposition & Delegation

🖥️ Runs 100% Locally (No cloud dependency)

🤖 Powered by Ollama (Qwen 2.5)

📊 End-to-end Data Science Pipeline Support

🗂️ Persistent Session Memory (SQLite)

🛠️ Rich Tooling: Pandas, Python, CSV, Shell, Visualization, Web Search

🔐 Safe execution with approval-based file & code writes

🏗️ System Architecture
AgentOS
 └── Data Science Team (Team Leader)
 
      ├── CSV Loader Agent
      
      ├── File Manager Agent
      
      ├── Data Understanding (EDA) Agent
      
      ├── Data Cleaning Agent
      
      ├── Feature Engineering Agent
      
      ├── Visualization Agent
      
      ├── Coding Agent
      
      ├── Model Training Agent
      
      ├── Model Evaluation Agent
      
      ├── Experiment Tracking Agent
      
      ├── Error Debugging Agent
      
      ├── Code Review Agent
      
      └── Shell Agent


Each agent:

Has a well-defined role

Uses strictly scoped tools

Maintains conversation + session memory

Works under a central team leader that plans and delegates tasks autonomously

🧩 Agent Capabilities

📁 Data Handling

CSV discovery & controlled loading

File system navigation

Safe read/write policies

📊 Data Science

Autonomous exploratory data analysis

Statistical profiling & correlations

Missing value & outlier detection

Feature engineering & preprocessing

Visualization using matplotlib

🤖 Machine Learning

Model selection & training

Evaluation & comparison

Experiment tracking

Iterative improvement suggestions

🛠️ Engineering & Safety

Code generation with user approval

Error diagnosis & recovery

Code review & quality assurance

Controlled shell execution

🧠 Model & Runtime
LLM

Ollama

Model: qwen2.5

Runs locally at: http://localhost:11434

Memory

SQLite

Persistent session memory across runs

Stored in memory.db

📦 Requirements
System

Python 3.10+

Local machine (Windows / Linux / macOS)

Sufficient RAM for local LLMs

Ollama

Install and start Ollama:

ollama pull qwen2.5
ollama serve

Python Dependencies
pip install agno pandas numpy matplotlib duckduckgo-search python-dotenv


Tip: Use a virtual environment (.venv) for isolation.

📁 Project Structure
.
├── data/                     # CSV datasets

├── memory.db                 # SQLite agent memory

├── main.py                   # Entry point (this file)

├── .env                      # Environment variables

└── README.md


▶️ Running the System

CLI Mode (Recommended)

python main.py



This launches an interactive CLI interface where you can:

Ask questions

Load datasets

Run EDA

Train models

Debug errors

Iterate autonomously

🧪 Example Use Cases

“Load the CSV and analyze the dataset”

“Perform full EDA and summarize key insights”

“Clean the data and prepare it for modeling”

“Train multiple models and compare accuracy”

“Explain why the model is underfitting”

“Fix this error in my training pipeline”

“Create visualizations and save them”

🔒 Safety & Control Principles

❌ No file writes without approval

❌ No code execution without user consent

❌ No destructive shell commands

✅ Explicit reasoning & summaries

✅ Clear reporting of every step

🌱 Why This Project Matters

This is not just a chatbot.

It is:

A true agentic system

A team of specialists

A blueprint for local, private, autonomous AI workflows

A foundation for production-grade agent systems

📌 Future Enhancements

Web UI (Streamlit / FastAPI)

Model deployment agents

Versioned experiment dashboards

Multi-dataset project tracking

Cloud / hybrid execution modes

🧠 Philosophy

“Don’t ask one LLM to do everything.
Build a team, give them tools, memory, and autonomy.”

📜 License

MIT License — free to use, modify, and extend.
