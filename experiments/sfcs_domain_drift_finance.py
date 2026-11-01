import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# MODEL
# ==========================================================

print("Loading embedding model...")

MODEL = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================================
# REFERENCE FIELD
# ==========================================================

REFERENCE_FIELD = [

    "Corporate finance evaluates capital allocation decisions.",
    "Investment analysis examines risk and return.",
    "Financial valuation estimates company value.",
    "Portfolio management balances risk and performance.",
    "Cash flow analysis supports investment decisions."
]

# ==========================================================
# DOMAIN DRIFT LEVELS
# ==========================================================

DRIFT_LEVELS = {

    0: {
        "name": "Corporate Finance",
        "responses": [
            "Capital budgeting supports investment decisions.",
            "Financial valuation estimates firm value.",
            "Portfolio construction balances risk and return.",
            "Cash flow forecasting guides finance strategy.",
            "Investment analysis evaluates opportunities."
        ]
    },

    1: {
        "name": "Investing",
        "responses": [
            "Investors seek attractive returns.",
            "Portfolio diversification reduces risk.",
            "Market valuation influences decisions.",
            "Asset allocation affects performance.",
            "Long-term investing compounds returns."
        ]
    },

    2: {
        "name": "Economics",
        "responses": [
            "Economic growth affects markets.",
            "Inflation influences purchasing power.",
            "Interest rates affect investment behavior.",
            "Supply and demand shape prices.",
            "Economic cycles influence business activity."
        ]
    },

    3: {
        "name": "Macroeconomics",
        "responses": [
            "GDP growth measures economic activity.",
            "Monetary policy affects inflation.",
            "Central banks regulate interest rates.",
            "Employment figures indicate economic health.",
            "Fiscal policy influences demand."
        ]
    },

    4: {
        "name": "Political Economy",
        "responses": [
            "Governments influence economic outcomes.",
            "Public policy shapes market incentives.",
            "Regulation affects business activity.",
            "Trade policy impacts industries.",
            "Institutional structures shape economies."
        ]
    },

    5: {
        "name": "Politics",
        "responses": [
            "Political parties compete for power.",
            "Election campaigns influence voters.",
            "Government coalitions create legislation.",
            "Public policy reflects political priorities.",
            "Democratic systems enable representation."
        ]
    },

    6: {
        "name": "International Relations",
        "responses": [
            "Diplomacy reduces international tensions.",
            "Military alliances affect security.",
            "International organizations coordinate policy.",
            "Geopolitical conflicts influence stability.",
            "Foreign policy shapes relations."
        ]
    },

    7: {
        "name": "History",
        "responses": [
            "Historical events shape societies.",
            "Ancient civilizations developed trade.",
            "Empires expanded through conquest.",
            "Industrialization transformed economies.",
            "Historical records preserve knowledge."
        ]
    },

    8: {
        "name": "Random Topics",
        "responses": [
            "Cats are popular pets.",
            "Volcanoes produce lava.",
            "Pasta is common in Italian cuisine.",
            "Rainfall supports agriculture.",
            "Mountains form through tectonic activity."
        ]
    }
}


# ==========================================================
# SFCS CALCULATION
# ==========================================================

def calculate_sfcs(reference_responses,
                   current_responses):

    reference_embedding = MODEL.encode(
        [" ".join(reference_responses)]
    )

    current_embeddings = MODEL.encode(
        current_responses
    )

    similarities = cosine_similarity(
        reference_embedding,
        current_embeddings
    )

    sfcs = float(
        np.mean(similarities)
    )

    variance = float(
        np.std(similarities)
    )

    return sfcs, variance


# ==========================================================
# RUN DRIFT ANALYSIS
# ==========================================================

levels = []
labels = []
scores = []
variances = []

print("\n")
print("=" * 60)
print("SFCS DOMAIN DRIFT SUITE")
print("=" * 60)

for level, info in DRIFT_LEVELS.items():

    sfcs, variance = calculate_sfcs(
        REFERENCE_FIELD,
        info["responses"]
    )

    levels.append(level)
    labels.append(info["name"])
    scores.append(sfcs)
    variances.append(variance)

    print(
        f"Level {level} | "
        f"{info['name']:<25} "
        f"SFCS={sfcs:.3f} "
        f"Var={variance:.3f}"
    )

print("=" * 60)

# ==========================================================
# DRIFT DETECTION
# ==========================================================

print("\nDRIFT ANALYSIS\n")

for i in range(1, len(scores)):

    drift = scores[i - 1] - scores[i]

    print(
        f"{labels[i - 1]} -> "
        f"{labels[i]} | "
        f"Drift={drift:.3f}"
    )

# ==========================================================
# FIGURE 1
# DRIFT CURVE
# ==========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    levels,
    scores,
    marker="o",
    linewidth=3,
    markersize=8
)

for x, y in zip(
        levels,
        scores):
    plt.annotate(
        f"{y:.3f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center"
    )

plt.xlabel(
    "Drift Level"
)

plt.ylabel(
    "Semantic Field Coherence Score"
)

plt.title(
    "Progressive Domain Drift Simulation"
)

plt.grid(True)

plt.tight_layout()

# ==========================================================
# FIGURE 2
# BAR CHART
# ==========================================================

plt.figure(figsize=(12, 6))

bars = plt.bar(
    labels,
    scores
)

for bar, score in zip(
        bars,
        scores):

    plt.text(
        bar.get_x() + bar.get_width()/2,
        score + 0.02,
        f"{score:.2f}",
        ha="center"
    )

plt.ylabel(
    "SFCS"
)

plt.title(
    "Semantic Field Coherence Across Drift Levels"
)

plt.xticks(rotation=45)

plt.tight_layout()

# ==========================================================
# FIGURE 3
# VARIANCE
# ==========================================================

plt.figure(figsize=(12, 6))

plt.bar(
    labels,
    variances
)

plt.ylabel(
    "Variance"
)

plt.title(
    "Variance Across Drift Levels"
)

plt.xticks(rotation=45)

plt.tight_layout()

# ==========================================================
# SHOW
# ==========================================================

plt.show()