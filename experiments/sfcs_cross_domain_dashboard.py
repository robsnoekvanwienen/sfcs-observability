import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# CROSS DOMAIN RESULTS
# ==========================================================

finance = [
    0.572,
    0.412,
    0.247,
    0.175,
    0.196,
    0.067,
    0.086,
    0.009,
    -0.022
]

ai = [
    0.522,
    0.183,
    0.133,
    0.111,
    0.121,
    0.032,
    0.027
]

medical = [
    0.554,
    0.310,
    0.016,
    0.021,
    0.059,
    0.047,
    0.023
]

legal = [
    0.522,
    0.414,
    0.184,
    0.066,
    -0.023
]

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("SFCS CROSS-DOMAIN VALIDATION")
print("=" * 60)

domains = {
    "Finance": finance,
    "AI": ai,
    "Medical": medical,
    "Legal": legal
}

for name, scores in domains.items():

    start = scores[0]
    end = scores[-1]

    print(
        f"{name:<10} "
        f"Start={start:.3f} "
        f"End={end:.3f} "
        f"Drop={start-end:.3f}"
    )

print("=" * 60)

# ==========================================================
# FIGURE 1
# CROSS-DOMAIN CURVES
# ==========================================================

plt.figure(figsize=(14, 8))

plt.plot(
    range(len(finance)),
    finance,
    marker="o",
    linewidth=3,
    label="Finance"
)

plt.plot(
    range(len(ai)),
    ai,
    marker="o",
    linewidth=3,
    label="AI"
)

plt.plot(
    range(len(medical)),
    medical,
    marker="o",
    linewidth=3,
    label="Medical"
)

plt.plot(
    range(len(legal)),
    legal,
    marker="o",
    linewidth=3,
    label="Legal"
)

plt.title(
    "Cross-Domain SFCS Validation"
)

plt.xlabel(
    "Progressive Drift Level"
)

plt.ylabel(
    "Semantic Field Coherence Score (SFCS)"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

# ==========================================================
# FIGURE 2
# START VS END COMPARISON
# ==========================================================

plt.figure(figsize=(10, 6))

domain_names = [
    "Finance",
    "AI",
    "Medical",
    "Legal"
]

start_scores = [
    finance[0],
    ai[0],
    medical[0],
    legal[0]
]

end_scores = [
    finance[-1],
    ai[-1],
    medical[-1],
    legal[-1]
]

x = np.arange(len(domain_names))

width = 0.35

plt.bar(
    x - width/2,
    start_scores,
    width,
    label="Reference Field"
)

plt.bar(
    x + width/2,
    end_scores,
    width,
    label="Random Topics"
)

plt.xticks(
    x,
    domain_names
)

plt.ylabel("SFCS")

plt.title(
    "Reference Field vs Random Topics"
)

plt.legend()

plt.tight_layout()

# ==========================================================
# FIGURE 3
# TOTAL DEGRADATION
# ==========================================================

plt.figure(figsize=(10, 6))

total_drop = [
    finance[0] - finance[-1],
    ai[0] - ai[-1],
    medical[0] - medical[-1],
    legal[0] - legal[-1]
]

bars = plt.bar(
    domain_names,
    total_drop
)

for bar, value in zip(
        bars,
        total_drop):

    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.ylabel(
    "Total SFCS Drop"
)

plt.title(
    "Semantic Field Degradation Across Domains"
)

plt.tight_layout()

# ==========================================================
# FIGURE 4
# NORMALIZED COMPARISON
# ==========================================================

plt.figure(figsize=(12, 7))

def normalize(scores):

    start = scores[0]

    return [s / start for s in scores]

plt.plot(
    normalize(finance),
    marker="o",
    linewidth=3,
    label="Finance"
)

plt.plot(
    normalize(ai),
    marker="o",
    linewidth=3,
    label="AI"
)

plt.plot(
    normalize(medical),
    marker="o",
    linewidth=3,
    label="Medical"
)

plt.plot(
    normalize(legal),
    marker="o",
    linewidth=3,
    label="Legal"
)

plt.title(
    "Normalized Cross-Domain Drift Curves"
)

plt.xlabel(
    "Progressive Drift"
)

plt.ylabel(
    "Normalized SFCS"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

# ==========================================================
# OPTIONAL EXPORTS
# ==========================================================

# Uncomment if desired

# plt.savefig(
#     "cross_domain_validation.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# ==========================================================
# SHOW
# ==========================================================

plt.show()