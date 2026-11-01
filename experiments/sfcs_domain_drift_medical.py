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

    "Cardiology studies heart function and cardiovascular disease.",
    "Cardiologists diagnose and treat cardiac disorders.",
    "Heart health depends on cardiovascular function.",
    "Cardiac imaging supports diagnosis of heart disease.",
    "Blood circulation is central to cardiovascular medicine."
]

# ==========================================================
# MEDICAL DRIFT LEVELS
# ==========================================================

DRIFT_LEVELS = {

    0: {
        "name": "Cardiology",
        "responses": [
            "Heart disease affects cardiovascular function.",
            "Cardiac imaging helps diagnose heart disorders.",
            "Blood pressure influences heart health.",
            "Cardiologists treat cardiovascular disease.",
            "Coronary arteries supply blood to the heart."
        ]
    },

    1: {
        "name": "Medicine",
        "responses": [
            "Physicians diagnose and treat illness.",
            "Medical imaging supports clinical decisions.",
            "Patient care requires evidence-based practice.",
            "Disease prevention improves public health.",
            "Treatment plans depend on diagnosis."
        ]
    },

    2: {
        "name": "Biology",
        "responses": [
            "Cells are the basic units of life.",
            "Genes influence biological traits.",
            "Evolution explains adaptation.",
            "Proteins perform cellular functions.",
            "Organisms interact within ecosystems."
        ]
    },

    3: {
        "name": "Chemistry",
        "responses": [
            "Atoms form chemical compounds.",
            "Chemical reactions transform substances.",
            "Molecules consist of bonded atoms.",
            "Organic chemistry studies carbon compounds.",
            "Thermodynamics influences reaction behavior."
        ]
    },

    4: {
        "name": "Physics",
        "responses": [
            "Newtonian mechanics describes motion.",
            "Electromagnetism governs electrical phenomena.",
            "Energy is conserved in physical systems.",
            "Quantum theory explains microscopic behavior.",
            "Relativity links space and time."
        ]
    },

    5: {
        "name": "History",
        "responses": [
            "Ancient civilizations developed agriculture.",
            "Historical events shaped modern societies.",
            "Empires expanded through conquest.",
            "Industrialization transformed economies.",
            "Historical records preserve knowledge."
        ]
    },

    6: {
        "name": "Random Topics",
        "responses": [
            "Cats are common household pets.",
            "Volcanoes release lava.",
            "Pasta is popular in Italy.",
            "Rainfall supports crops.",
            "Mountains form through tectonic activity."
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

    sfcs = float(np.mean(similarities))
    variance = float(np.std(similarities))

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
print("SFCS MEDICAL DOMAIN DRIFT SUITE")
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
plt.title("Medical Domain Drift Simulation")
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
    "Semantic Field Coherence Across Medical Drift Levels"
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
    "Variance Across Medical Drift Levels"
)

plt.xticks(rotation=45)

plt.tight_layout()

# ==========================================================
# SHOW FIGURES
# ==========================================================

plt.show()