from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.team import Team
from agno.os import AgentOS
from agno.db.sqlite import SqliteDb

from pathlib import Path

from agno.tools.csv_toolkit import CsvTools
from agno.tools.file import FileTools
from agno.tools.pandas import PandasTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools
from agno.tools.visualization import VisualizationTools

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# Model
# =============================================================================

llm = Ollama(
    id="qwen2.5",
    host="http://localhost:11434"
)

# =============================================================================
# Database
# =============================================================================

db = SqliteDb(
    db_file="memory.db",
    session_table="session_table"
)

# =============================================================================
# Agents
# =============================================================================

# -------------------- CSV Loader Agent --------------------

csv_loader_agent = Agent(
    id="data_loader_agent",
    name="data_loader_agent",
    role="CSV Data Loading Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    instructions=[
        "You are an expert data loading agent.",
        "Use file tools to list directories and identify CSV files.",
        "Always list available CSV files before reading any file.",
        "Never attempt to read a directory as a CSV file.",
        "Read only one CSV file at a time.",
        "Limit reading to at most 20–30 rows unless explicitly instructed otherwise.",
        "Use CSV tools exclusively for reading CSV files.",
        "If multiple CSV files exist, ask the user which one to read."
    ],
    tools=[
        CsvTools(
            csvs=list(DATA_DIR.glob("*.csv")),
            enable_query_csv_file=False
        ),
        FileTools(base_dir=BASE_DIR)
    ]
)

# -------------------- File Manager Agent --------------------

file_manager_agent = Agent(
    id="file_manager_agent",
    name="file_manager_agent",
    role="File System Manager",
    model=llm,
    instructions=[
        "You are an expert file management agent.",
        "List project files and directories when requested.",
        "You may read and write non-CSV files.",
        "Never read CSV files."
    ],
    tools=[FileTools(base_dir=BASE_DIR)]
)

# -------------------- Data Cleaning Agent --------------------

data_cleaning_agent = Agent(
    id="data_cleaning_agent",
    name="data_cleaning_agent",
    role="Data Cleaning and Preprocessing Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "You are an expert data cleaning and preprocessing agent.",
        "Prepare raw datasets for modeling.",
        "Identify missing values and justify handling strategies.",
        "Detect and remove duplicate rows while reporting counts.",
        "Fix incorrect data types and convert numeric strings when required.",
        "Detect basic outliers using simple statistical methods.",
        "Do not remove outliers unless explicitly instructed.",
        "Do not perform exploratory analysis or visualization.",
        "Do not overwrite raw files without user approval.",
        "Clearly summarize all cleaning steps."
    ],
    tools=[PandasTools(), FileTools(base_dir=BASE_DIR)]
)

# -------------------- Data Understanding Agent --------------------

data_understanding_agent = Agent(
    id="data_understanding_agent",
    name="data_understanding_agent",
    role="Autonomous Exploratory Data Analysis Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        # Role & Autonomy
        "You are a senior data scientist specialized in autonomous exploratory data analysis.",
        "Given a dataset, independently decide which analyses are required without asking the user.",
        "Actively call Python tools to compute statistics, not just describe them conceptually.",

        # Dataset Inspection
        "Load the dataset and report dataset shape, column names, and data types.",
        "Identify numerical, categorical, datetime, and boolean features.",
        "Detect target variable if inferable from context or naming.",

        # Data Quality Checks
        "Compute missing value counts and percentages per column.",
        "Identify duplicate rows and constant or near-constant columns.",
        "Detect potential data leakage features.",
        "Check for incorrect data types and suspicious values.",

        # Statistical Analysis
        "For numerical features, compute mean, median, std, min, max, skewness, and kurtosis.",
        "Detect outliers using IQR and z-score methods.",
        "For categorical features, compute frequency distributions and dominant classes.",
        "Check class imbalance if a target variable exists.",

        # Feature Relationships
        "Compute correlation matrix for numerical features.",
        "Identify highly correlated feature pairs and multicollinearity risks.",
        "If a target exists, compute feature–target relationships (correlation, group statistics).",
        "Highlight features with strongest predictive potential.",

        # Distribution Analysis
        "Analyze distributions (normal, skewed, multimodal) and recommend transformations.",
        "Suggest scaling or encoding strategies where appropriate.",

        # Insights & Reasoning
        "Synthesize findings into clear, data-driven insights.",
        "Explain what the data suggests about complexity, signal strength, and modeling difficulty.",
        "Explicitly state assumptions and uncertainties.",

        # Output Discipline
        "Structure output as: Data Overview → Data Quality → Statistics → Relationships → Key Insights.",
        "Avoid generic statements; every insight must be supported by computed evidence."
    ],
    tools=[PandasTools(),PythonTools, FileTools(base_dir=BASE_DIR)]
)

# -------------------- Feature Engineering Agent --------------------

feature_engineering_agent = Agent(
    id="feature_engineering_agent",
    name="feature_engineering_agent",
    role="Feature Engineering Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "You are a feature engineering specialist.",
        "Use pandas and python tools to transform cleaned datasets into model-ready features.",
        "Create meaningful derived features with justification.",
        "Apply appropriate encoding strategies for categorical variables.",
        "Scale numerical features when required.",
        "Never scale target variables unless instructed.",
        "Remove redundant features only when justified.",
        "Do not perform EDA or model training.",
        "Do not overwrite datasets without approval.",
        "Give proper insights and descriptions of your created model ready features",
        "Give a valuable conclusion of your work"
    ],
    tools=[PandasTools(),PythonTools(base_dir=BASE_DIR), FileTools(base_dir=BASE_DIR)]
)

# -------------------- Visualization Agent --------------------

visualization_agent = Agent(
    id="visualization_agent",
    name="visualization_agent",
    role="Data Visualization Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "You are an expert in data visualization using matplotlib.",
        "Create bar plots, histograms, line plots, scatter plots, and pie charts using Visualization and python tools.",
        "Use appropriate plots based on feature types.",
        "Do not modify datasets.",
        "Give proper explainations of your created plots and valuable insights.",
        "Use the python tools to create the plots and show it to the user.",
        "once the user verifies the plots use the file tools to save the plots in the base directory"
    ],
    tools=[
        PandasTools(),PythonTools(base_dir=BASE_DIR),FileTools(base_dir=BASE_DIR),
        FileTools(base_dir=BASE_DIR),
        VisualizationTools("plots")
    ]
)

# -------------------- Coding Agent --------------------

coding_agent = Agent(
    id="coding_agent",
    name="coding_agent",
    role="Python Machine Learning Developer",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    read_chat_history=True,
    instructions=[
        "You are an expert Python and ML developer.",
        "Write clean and modular machine learning code.",
        "Use numpy, pandas, scipy, and sklearn where appropriate.",
        "Never write or execute code without user approval.",
        "Use shell tools only to install packages using 'uv run <package>'."
        ,"Always show the code drafted by you to the user in chat get it reviewed",
        "You can search the web to get latest coding related information",
        "Once your code is verified by a user use the file tools to create a .py file and write and save the code in that file"
    ],
    tools=[
        DuckDuckGoTools(),
        PythonTools(base_dir=BASE_DIR),
        FileTools(base_dir=BASE_DIR),
        ShellTools(base_dir=BASE_DIR)
    ]
)

# -------------------- Model Training Agent --------------------

model_training_agent = Agent(
    id="model_training_agent",
    name="model_training_agent",
    role="Model Training and Evaluation Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "Train and evaluate machine learning models.",
        "Perform reproducible train-validation splits.",
        "Always use multiple models to get the perfect and most accurate results possible",
        "Select appropriate models and metrics.",
        "Generate training code only after user approval.",
        "Do not perform EDA or feature engineering.",
        "Do not deploy models."
    ],
    tools=[PandasTools(), PythonTools(base_dir=BASE_DIR), FileTools(base_dir=BASE_DIR)]
)

# -------------------- Experiment Tracking Agent --------------------

experiment_tracking_agent = Agent(
    id="experiment_tracking_agent",
    name="experiment_tracking_agent",
    role="Experiment Tracking and Metadata Manager",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "Track machine learning experiments and metadata.",
        "Store datasets, features, models, metrics, and timestamps.",
        "Retrieve and compare past experiments.",
        "Never modify datasets or models."
    ],
    tools=[PythonTools(base_dir=BASE_DIR), FileTools(base_dir=BASE_DIR), PandasTools()]
)

# -------------------- Model Evaluation Agent --------------------

model_evaluation_agent = Agent(
    id="model_evaluation_agent",
    name="model_evaluation_agent",
    role="Model Evaluation and Comparison Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "Compare trained models using appropriate metrics.",
        "If the accuracy is low suggest the coding agent more accurate models and iterate through the steps to get more accurate model",
        "Detect overfitting and underfitting.",
        "Rank models and explain trade-offs.",
        "Suggest concrete improvements."
    ],
    tools=[PandasTools(),PythonTools(base_dir=BASE_DIR),FileTools(base_dir=BASE_DIR)]
)

# -------------------- Error Debugging Agent --------------------

error_debugging_agent = Agent(
    id="error_debugging_agent",
    name="error_debugging_agent",
    role="Error Debugging and Recovery Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "Analyze errors and stack traces.",
        "Explain root causes clearly.",
        "Propose minimal and safe fixes.",
        "Never modify files without approval."
    ],
    tools=[PythonTools(base_dir=BASE_DIR), FileTools(base_dir=BASE_DIR)]
)

# -------------------- Code Review Agent --------------------

code_review_agent = Agent(
    id="code_review_agent",
    name="code_review_agent",
    role="Code Review and Quality Assurance Specialist",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "Review code for correctness, clarity, and maintainability.",
        "Check for data leakage and reproducibility issues.",
        "Verify APIs and best practices using web search if needed.",
        "Never execute or modify code.",
        "Explicitly approve code if no major issues are found."
    ],
    tools=[FileTools(base_dir=BASE_DIR), DuckDuckGoTools()]
)

# -------------------- Shell Agent --------------------

shell_agent = Agent(
    id="shell_agent",
    name="shell_agent",
    role="Shell Command Executor",
    model=llm,
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    search_session_history=True,
    instructions=[
        "Execute shell commands with extreme caution.",
        "Only run Python files using 'uv run <python_file>'.",
        "Never modify project structure or delete files."
    ],
    tools=[ShellTools(base_dir=BASE_DIR)]
)

# =============================================================================
# Team
# =============================================================================

data_science_team = Team(
    id="data_science_team",
    name="data-science-team",
    model=llm,
    db=db,
    role="Team Leader and Project Manager",
    members=[
        csv_loader_agent,
        file_manager_agent,
        data_understanding_agent,
        data_cleaning_agent,
        feature_engineering_agent,
        visualization_agent,
        coding_agent,
        model_training_agent,
        experiment_tracking_agent,
        model_evaluation_agent,
        error_debugging_agent,
        code_review_agent,
        shell_agent
    ],
    add_history_to_context=True,
    add_member_tools_to_context=True,
    enable_agentic_state=True,
    session_state={},
    instructions=[
        "You are an expert data science team leader.",
        "You have the knowledge of all the capability and roles of your agents."
        "Break complex tasks into smaller steps.",
        "You have appropriate tools to perform all the ML and Data science task to create a professional ML pipeline.So always try to give ur own suggestions ,ideas,plans and vision and perfom and delegate the appropriate  task freely without wasting much time."
        "Delegate tasks internally without exposing delegation.",
        "Collect and summarize outputs and results achieved my each agent for the user.",
        "If any agent faces any error use ur debugger agent to fix the error and get valid outputs",
        "Request user approval before modifying files or running code.",
        "Maintain session memory for important project details."
    ]
)

# =============================================================================
# Agent OS
# =============================================================================

agent_os = AgentOS(
    id="agent_os",
    name="Data Science Team",
    description="A multi-agent system to guide users through end-to-end data science workflows.",
    teams=[data_science_team]
)

app = agent_os.get_app()

if __name__ == "__main__":
    data_science_team.cli_app()
