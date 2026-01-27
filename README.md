# group-penalized-tfidf

**Overview**
This repository implements a group-penalized TF-IDF weighting scheme designed for think-aloud and cognitive reasoning transcript data. The method reduces item-specific lexical effects while preserving cross-item reasoning language, improving generalization for downstream machine-learning tasks such as strategy classification and clustering.

The approach is motivated by a common failure mode of standard TF-IDF in cognitive task data: terms that are discriminative within a specific item or problem are often over-weighted, even when they do not reflect stable reasoning strategies.

**Motivation**
In think-aloud protocols and related cognitive datasets, transcripts are often structured around multiple items, tasks, or problems. Standard TF-IDF can unintentionally emphasize item-specific vocabulary (e.g., surface features of a particular problem), leading models to learn item artifacts rather than underlying cognitive strategies.

This reduces:

Cross-item generalization

Interpretability of learned features

Transferability of models to new tasks

The goal of group-penalized TF-IDF is to explicitly encode cross-item stability as an inductive bias in the feature representation.

**Methods Overview**
At a high level, the method modifies standard TF-IDF by incorporating group-level information (e.g., task or item identifiers):

Documents are associated with a known group structure (e.g., problem ID).

Term weights are penalized when their discriminative power is largely confined to a single group.

Terms that appear consistently across groups retain higher weight, reflecting more general reasoning language.

This produces feature representations that are less sensitive to item-specific lexical artifacts.

**Mininal Example in R**
tfidf_matrix <- group_penalized_tfidf(
  documents = documents,
  groups = groups
)

**Limitations**
This approach makes several assumptions:

Group structure (e.g., item or task IDs) must be known.

The method does not capture semantic similarity beyond surface lexical patterns.

Rare but meaningful item-specific terms may be underweighted.

Not intended as a replacement for embedding-based models when semantic abstraction is the primary goal.
