import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# MODEL
# =====================================

print("Loading embedding model...")

MODEL = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =====================================
# SCENARIOS
# =====================================

SCENARIOS = {

    "A": {
        "name": "Highly Stable",
        "responses": [
            "Overfitting occurs when a model learns noise instead of general patterns.",
            "Overfitting happens when a model captures noise rather than generalizing.",
            "A model overfits when it memorizes training data instead of learning general patterns.",
            "Overfitting means learning noise from training data instead of broader patterns.",
            "An overfitted model memorizes noise rather than learning generalizable structure."
        ]
    },

    "B": {
        "name": "Stable",
        "responses": [
            "Overfitting occurs when a model learns noise.",
            "Regularization helps reduce overfitting.",
            "Cross-validation can detect overfitting.",
            "Overfitted models perform poorly on unseen data.",
            "Increasing training data can reduce overfitting."
        ]
    },

    "C": {
        "name": "Degrading",
        "responses": [
            "Overfitting occurs when a model learns noise.",
            "Neural networks contain hidden layers.",
            "Machine learning requires training data.",
            "Gradient descent minimizes error.",
            "Feature engineering can improve performance."
        ]
    },

    "D": {
        "name": "Unstable",
        "responses": [
            "Cats are popular pets.",
            "The stock market fluctuates daily.",
            "Mount Everest is the highest mountain.",
            "Pasta is a traditional Italian dish.",
            "Rainfall influences agriculture."
        ]
    }
}


# =====================================
# CALCULATE SFCS
# =====================================

def calculate_rcs(responses):

    embeddings = MODEL.encode(responses)

    matrix = cosine_similarity(
        embeddings
    )

    similarities = []

    n = len(matrix)

    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(matrix[i][j])

    sfcs = np.mean(similarities)

    variance = np.std(similarities)

    return sfcs, variance, matrix


# =====================================
# CLASSIFICATION
# =====================================

def classify(sfcs):

    if sfcs >= 0.90:
        return "HIGHLY STABLE"

    elif sfcs >= 0.80:
        return "STABLE"

    elif sfcs >= 0.70:
        return "ACCEPTABLE"

    elif sfcs >= 0.60:
        return "DEGRADING"

    else:
        return "UNSTABLE"


# =====================================
# HEATMAP
# =====================================

def display_heatmap(matrix, scenario_name):

    plt.figure(figsize=(6, 5))

    plt.imshow(
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=1
    )

    plt.colorbar(label="Similarity")

    plt.title(
        f"SFCS Similarity Matrix\n{scenario_name}"
    )

    plt.xlabel("Run")
    plt.ylabel("Run")

    plt.tight_layout()

    plt.show()


# =====================================
# MAIN MENU
# =====================================

print("\n")
print("=" * 45)
print("SFCS VALIDATION WORKBENCH V1")
print("=" * 45)

print("\nAvailable Scenarios:\n")

for key, value in SCENARIOS.items():
    print(f"{key} - {value['name']}")

choice = input(
    "\nSelect scenario (A-D): "
).strip().upper()

if choice not in SCENARIOS:
    print("\nInvalid selection.")
    exit()

scenario = SCENARIOS[choice]

responses = scenario["responses"]

print("\nRunning analysis...\n")

sfcs, variance, matrix = calculate_rcs(
    responses
)

status = classify(sfcs)

print("=" * 45)
print("RESULTS")
print("=" * 45)

print(
    f"Scenario     : {scenario['name']}"
)

print(
    f"SFCS Score    : {sfcs:.3f}"
)

print(
    f"Variance     : {variance:.3f}"
)

print(
    f"Classification: {status}"
)

print("=" * 45)

display_heatmap(
    matrix,
    scenario["name"]
)