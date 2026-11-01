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
# VALIDATION V2 SCENARIOS
# =====================================

SCENARIOS = {

    "E": {
        "name": "Synonyms Test",
        "responses": [
            "The model should be monitored.",
            "Continuous observation is recommended.",
            "Regular oversight is advised.",
            "Ongoing monitoring is suggested.",
            "The system requires continued supervision."
        ]
    },

    "F": {
        "name": "Same Conclusion Different Reasoning",
        "responses": [
            "The company should invest because revenue is increasing.",
            "The company should invest because competitors are expanding.",
            "The company should invest because demand is growing.",
            "The company should invest because innovation is accelerating.",
            "The company should invest because market conditions are favorable."
        ]
    },

    "G": {
        "name": "Domain Shift",
        "responses": [
            "Machine learning models require training data.",
            "Data science combines statistics and computing.",
            "Statistical inference estimates population properties.",
            "Predictive analytics uses historical data.",
            "Feature engineering improves model performance."
        ]
    },

    "H": {
        "name": "Contradictory Statements",
        "responses": [
            "Overfitting is harmful.",
            "Overfitting is beneficial.",
            "Overfitting reduces generalization.",
            "Overfitting improves generalization.",
            "Overfitting can damage predictive accuracy."
        ]
    },

    "I": {
        "name": "Near Identical",
        "responses": [
            "Overfitting occurs when a model learns noise.",
            "Overfitting occurs when a model learns noise.",
            "Overfitting occurs when a model learns noise.",
            "Overfitting occurs when a model learns noise.",
            "Overfitting occurs when a model learns noise."
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
            similarities.append(
                matrix[i][j]
            )

    sfcs = np.mean(similarities)

    variance = np.std(similarities)

    return sfcs, variance, matrix


# =====================================
# HEATMAP
# =====================================

def display_heatmap(matrix, title):

    plt.figure(figsize=(6, 5))

    plt.imshow(
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=1
    )

    plt.colorbar(label="Similarity")

    plt.title(
        f"SFCS Validation V2\n{title}"
    )

    plt.xlabel("Run")
    plt.ylabel("Run")

    plt.tight_layout()

    plt.show()


# =====================================
# EXPECTATIONS
# =====================================

EXPECTATIONS = {

    "E": "Expected: Very High SFCS",
    "F": "Expected: Medium-High SFCS",
    "G": "Expected: Medium SFCS",
    "H": "Expected: Potential Failure Case",
    "I": "Expected: Near Perfect SFCS"
}


# =====================================
# MAIN
# =====================================

print("\n")
print("=" * 55)
print("SFCS VALIDATION WORKBENCH V2")
print("=" * 55)

for key, value in SCENARIOS.items():

    print(
        f"{key} - {value['name']}"
    )

choice = input(
    "\nSelect scenario (E-I): "
).strip().upper()

if choice not in SCENARIOS:
    print("Invalid scenario.")
    exit()

scenario = SCENARIOS[choice]

responses = scenario["responses"]

print("\nRunning analysis...\n")

sfcs, variance, matrix = calculate_rcs(
    responses
)

print("=" * 55)
print("VALIDATION RESULTS")
print("=" * 55)

print(
    f"Scenario       : {scenario['name']}"
)

print(
    f"Expectation    : {EXPECTATIONS[choice]}"
)

print(
    f"SFCS            : {sfcs:.3f}"
)

print(
    f"Variance       : {variance:.3f}"
)

print("=" * 55)

# =====================================
# INTERPRETATION
# =====================================

if choice == "H":

    print("\nADVERSARIAL TEST")

    print(
        "If SFCS remains high despite contradictory "
        "semantic meaning, this reveals a limitation "
        "of semantic similarity as a proxy for "
        "semantic field coherence."
    )

elif choice == "F":

    print("\nREASONING TEST")

    print(
        "This experiment evaluates whether SFCS "
        "tracks final conclusions or underlying "
        "reasoning pathways."
    )

elif choice == "G":

    print("\nDOMAIN TEST")

    print(
        "This experiment evaluates how SFCS behaves "
        "when responses remain in a related domain "
        "but discuss different concepts."
    )

display_heatmap(
    matrix,
    scenario["name"]
)