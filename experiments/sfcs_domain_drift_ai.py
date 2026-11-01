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

    "Machine learning models learn patterns from data.",
    "Supervised learning uses labeled training examples.",
    "Model generalization determines performance on unseen data.",
    "Overfitting occurs when models learn noise instead of patterns.",
    "Feature engineering improves predictive performance."
]

# ==========================================================
# DOMAIN DRIFT LEVELS
# ==========================================================

DRIFT_LEVELS = {

    0: {
        "name": "Machine Learning",
        "responses": [
            "Neural networks learn patterns from data.",
            "Training data is required for supervised learning.",
            "Feature engineering improves model performance.",
            "Cross validation evaluates generalization.",
            "Overfitting reduces performance on unseen data."
        ]
    },

    1: {
        "name": "Statistics",
        "responses": [
            "Statistical inference estimates population properties.",
            "Probability distributions describe uncertainty.",
            "Hypothesis testing evaluates scientific claims.",
            "Regression analysis identifies relationships.",
            "Sampling methods influence estimation quality."
        ]
    },

    2: {
        "name": "Mathematics",
        "responses": [
            "Linear algebra studies vector spaces.",
            "Calculus describes continuous change.",
            "Optimization identifies extrema of functions.",
            "Probability theory models uncertainty.",
            "Graph theory analyzes networks and connections."
        ]
    },

    3: {
        "name": "Physics",
        "responses": [
            "Newtonian mechanics describes motion.",
            "Quantum mechanics models microscopic systems.",
            "Thermodynamics studies energy transfer.",
            "Electromagnetism explains electrical phenomena.",
            "Relativity links space and time."
        ]
    },

    4: {
        "name": "Biology",
        "responses": [
            "Cells are the basic units of life.",
            "Evolution explains species adaptation.",
            "Genetics studies inheritance mechanisms.",
            "Proteins perform biological functions.",
            "Ecosystems contain interacting organisms."
        ]
    },

    5: {
        "name": "History",
        "responses": [
            "Ancient civilizations developed trade networks.",
            "Empires expanded through conquest.",
            "Industrialization transformed societies.",
            "Historical events influence modern institutions.",
            "Archives preserve historical information."
        ]
    },

    6: {
        "name": "Random Topics",
        "responses": [
            "Cats are common household pets.",
            "Volcanoes release magma from the Earth.",
            "Pasta is widely consumed in Italy.",
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
print("SFCS AI DOMAIN DRIFT SUITE")
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
        f"{info['name']:<20} "
        f"SFCS={sfcs:.3f} "
        f"Var={variance:.3f}"
    )

print("=" * 60)

# ==========================================================
# DRIFT ANALYSIS
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

for x, y in zip(levels, scores):

    plt.annotate(
        f"{y:.3f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center"
    )

plt.xlabel("Drift Level")
plt.ylabel("SFCS")
plt.title("AI Domain Drift Simulation")
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
        score + 0.01,
        f"{score:.2f}",
        ha="center"
    )

plt.ylabel("SFCS")
plt.title(
    "Semantic Field Coherence Across AI Drift Levels"
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

plt.ylabel("Variance")

plt.title(
    "Variance Across AI Drift Levels"
)

plt.xticks(rotation=45)

plt.tight_layout()

# ==========================================================
# EXPORT RESULTS
# ==========================================================

print("\nRESULTS\n")

for label, score in zip(labels, scores):

    print(
        f"{label:<20} "
        f"{score:.3f}"
    )

# ==========================================================
# SHOW
# ==========================================================

plt.show()