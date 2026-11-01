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

    "Contract law governs agreements between parties.",
    "Legal contracts define obligations and responsibilities.",
    "Courts interpret contractual disputes.",
    "Contract law ensures enforceable agreements.",
    "Legal remedies address breaches of contract."
]

# ==========================================================
# LEGAL DRIFT LEVELS
# ==========================================================

DRIFT_LEVELS = {

    0: {
        "name": "Contract Law",
        "responses": [
            "Contracts create enforceable obligations.",
            "Breach of contract may result in damages.",
            "Legal agreements require mutual consent.",
            "Courts interpret contractual clauses.",
            "Contract law protects parties."
        ]
    },

    1: {
        "name": "Law",
        "responses": [
            "Legal systems establish rules.",
            "Courts interpret legislation.",
            "Judges resolve disputes.",
            "Statutes guide legal decisions.",
            "Legal rights protect individuals."
        ]
    },

    2: {
        "name": "Politics",
        "responses": [
            "Political parties compete for influence.",
            "Governments create public policy.",
            "Elections determine representation.",
            "Legislation reflects political priorities.",
            "Democratic institutions shape governance."
        ]
    },

    3: {
        "name": "History",
        "responses": [
            "Historical events influence societies.",
            "Empires shaped political structures.",
            "Industrialization altered economies.",
            "Ancient civilizations developed governance.",
            "Historical records preserve information."
        ]
    },

    4: {
        "name": "Random Topics",
        "responses": [
            "Cats are popular pets.",
            "Volcanoes erupt periodically.",
            "Pasta is common in Italian cuisine.",
            "Rainfall influences agriculture.",
            "Mountains form over geological time."
        ]
    }
}

# ==========================================================
# SFCS CALCULATION
# ==========================================================

def calculate_sfcs(reference_responses, current_responses):

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
# RUN ANALYSIS
# ==========================================================

levels = []
labels = []
scores = []
variances = []

print("\n")
print("=" * 60)
print("SFCS LEGAL DOMAIN DRIFT SUITE")
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
# RESULTS SUMMARY
# ==========================================================

print("\nRESULTS\n")

for label, score in zip(labels, scores):

    print(
        f"{label:<20} "
        f"{score:.3f}"
    )

# ==========================================================
# FIGURE 1 - DRIFT CURVE
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

plt.title(
    "Legal Domain Drift Simulation"
)

plt.grid(True)

plt.tight_layout()

# ==========================================================
# FIGURE 2 - BAR CHART
# ==========================================================

plt.figure(figsize=(12, 6))

bars = plt.bar(
    labels,
    scores
)

for bar, score in zip(bars, scores):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.01,
        f"{score:.2f}",
        ha="center"
    )

plt.ylabel("SFCS")

plt.title(
    "Semantic Field Coherence Across Legal Drift Levels"
)

plt.xticks(rotation=45)

plt.tight_layout()

# ==========================================================
# FIGURE 3 - VARIANCE
# ==========================================================

plt.figure(figsize=(12, 6))

plt.bar(
    labels,
    variances
)

plt.ylabel("Variance")

plt.title(
    "Variance Across Legal Drift Levels"
)

plt.xticks(rotation=45)

plt.tight_layout()

# ==========================================================
# SHOW FIGURES
# ==========================================================

plt.show()