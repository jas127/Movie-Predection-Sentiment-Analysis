"""Central configuration: file paths and shared constants.

All paths are resolved relative to the project root so the app runs the same
way locally and on Streamlit Community Cloud (no hard-coded absolute paths).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
REPORTS_DIR = PROJECT_ROOT / "reports"

BOX_OFFICE_PATH = DATA_DIR / "box_office_data.csv"
REVIEWS_PATH = DATA_DIR / "moviereviews.tsv"

# --- Box office schema (as found in the original CSV) -----------------------
BOX_OFFICE_REQUIRED_COLUMNS = [
    "Rank", "Release Group", "$Worldwide", "$Domestic", "Domestic %",
    "$Foreign", "Foreign %", "Year", "Genres", "Rating", "Vote_Count",
    "Original_Language", "Production_Countries",
]
REVIEWS_REQUIRED_COLUMNS = ["label", "review"]

# --- Original revenue model definition (from Movie-Revenue-Prediction.ipynb) --
TARGET = "$Worldwide"
NUMERICAL_FEATURES = ["Year", "Vote_Count"]
CATEGORICAL_FEATURES = ["Genres", "Rating", "Original_Language", "Production_Countries"]
MODEL_FEATURES = ["Year", "Genres", "Rating", "Vote_Count", "Original_Language", "Production_Countries"]

# Comparison model that only uses information knowable before release.
# Vote_Count and Rating accumulate after release, so they are excluded.
PRE_RELEASE_NUMERICAL = ["Year"]
PRE_RELEASE_CATEGORICAL = ["Genres", "Original_Language", "Production_Countries"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# --- Sentiment ---------------------------------------------------------------
# Original notebook rule: compound >= 0 -> 'pos', else 'neg'.
ORIGINAL_CLASSIFICATION_THRESHOLD = 0.0
# Conventional VADER thresholds (Hutto & Gilbert) for 3-way display only.
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# Minimum number of films for a genre/group to be included in rankings,
# so tiny groups do not dominate "top" lists.
MIN_GROUP_SIZE = 30
