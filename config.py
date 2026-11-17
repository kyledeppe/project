
"""
STAT 636 Project
Targeted Vaccination on Real Contact Networks

Global configuration file.
Team:
    Kyle Deppe
    Tyler Lowry
    Madhav Thamaran
"""

from pathlib import Path

# --------------------------------------------------
# 1. PROJECT DIRECTORIES
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
SIMULATION_DIR = RESULTS_DIR / "simulations"

# Create directories automatically
for directory in [
    DATA_DIR,
    RESULTS_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    SIMULATION_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. DATA FILES
# --------------------------------------------------

DAY1_FILE = DATA_DIR / "day1.gexf"
DAY2_FILE = DATA_DIR / "day2.gexf"
METADATA_FILE = DATA_DIR / "metadata.csv"

# Expected network size
EXPECTED_NODES = 242

# Number of classrooms
EXPECTED_CLASSES = 10


# --------------------------------------------------
# 3. REPRODUCIBILITY
# --------------------------------------------------

RANDOM_SEED = 636


# --------------------------------------------------
# 4. VACCINATION PARAMETERS
# --------------------------------------------------

VACCINATION_BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20
]

VACCINATION_STRATEGIES = [
    "Random",
    "Degree",
    "Weighted_Degree",
    "Betweenness",
    "Community_Bridge",
    "Within_Community_Hub"
]


# --------------------------------------------------
# 5. SPECTRAL CLUSTERING PARAMETERS
# --------------------------------------------------

SPECTRAL_DIMENSIONS = 3

COMMUNITY_COUNTS = [
    5,
    8,
    10,
    12
]

DEFAULT_COMMUNITIES = 10

CLUSTERING_METHODS = [
    "KMeans",
    "GMM"
]


# --------------------------------------------------
# 6. EPIDEMIC PARAMETERS
# --------------------------------------------------

# Transmission probability rate per second
BETA_VALUES = [
    0.00005,
    0.0001,
    0.00025,
    0.0005,
]

# Duration of infectiousness in days
INFECTIOUS_PERIOD = 3

# Number of epidemic simulations
N_SIMULATIONS = 500

# Number of initial infection scenarios
N_INITIAL_CASES = 20


# --------------------------------------------------
# 7. NULL MODEL PARAMETERS
# --------------------------------------------------

NULL_MODELS = [
    "Erdos_Renyi",
    "Configuration",
    "DCSBM"
]

N_NULL_NETWORKS = 100


# --------------------------------------------------
# 8. STATISTICAL SETTINGS
# --------------------------------------------------

CONFIDENCE_LEVEL = 0.95

BOOTSTRAP_REPLICATES = 2000


# --------------------------------------------------
# 9. NETWORK SETTINGS
# --------------------------------------------------

WEIGHT_ATTRIBUTE = "duration"

# Contact duration is measured in seconds
CONTACT_INTERVAL = 20

# Inverse duration is used for shortest-path distances
MIN_EDGE_WEIGHT = 1e-10


# --------------------------------------------------
# 10. OUTPUT SETTINGS
# --------------------------------------------------

SAVE_CSV = True
SAVE_FIGURES = True
FIGURE_DPI = 300

print("STAT 636 configuration loaded.")
print(f"Project directory: {PROJECT_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Results directory: {RESULTS_DIR}")
