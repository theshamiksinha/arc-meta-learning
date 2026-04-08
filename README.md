# Meta-Learning on 1D-ARC: Few-Shot Sequence Transformation

Few-shot learning on the [1D Abstraction and Reasoning Corpus (1D-ARC)](https://github.com/khalil-research/1D-ARC) — a benchmark where a model must infer a transformation rule from 2–4 input/output examples and apply it to a new input it has never seen. Implemented and benchmarked four architectures progressing from a supervised BiLSTM baseline to a position-aware Conditional Neural Process (PosCNP) trained with MAML.

## The Problem

Each 1D-ARC task is a few-shot sequence transformation. Given a support set of 2–4 (input, output) sequence pairs, predict the output for a held-out query input. There are 18 concept classes (recolor, flip, move, fill, denoise, mirror, etc.) with 50 tasks each — 901 tasks total.

Two evaluation settings of increasing difficulty:

| Setting | Description |
|---|---|
| **Same-Concept** | 80/20 stratified split; test tasks share concept classes with training |
| **Cross-Concept** | Entire concept classes held out; model must generalize to unseen transformations |

## Models

| Model | Key Idea |
|---|---|
| **BiLSTM Baseline** | Supervised sequence-to-sequence; encodes support pairs independently |
| **BiLSTM + FOMAML** | Same architecture, trained with First-Order MAML for fast test-time adaptation |
| **Matching Networks** | Episodic training; attends over support embeddings to predict query output |
| **PosCNP** | Position-aware Conditional Neural Process; multi-head cross-attention from query onto support, preserving positional structure lost by CNP's mean-pooling |

All MAML-based models run 20 steps of SGD on the support set at test time before evaluating on the query.

**Data augmentation:** Token-color permutation — remaps non-zero token colors while preserving the underlying transformation structure.

## Results

### Same-Concept

| Model | No Adapt | + Adapt | + Aug |
|---|---|---|---|
| BiLSTM Baseline | 27.6% | 29.3% | 40.3% |
| BiLSTM + FOMAML | 30.4% | **48.6%** | 48.6% |
| Matching Networks | 55.3% | 60.8% | 60.8% |
| CNP | 52.5% | 72.9% | 72.9% |
| **PosCNP** | 61.3% | — | **79.0%** |

### Cross-Concept

| Model | Test Accuracy |
|---|---|
| BiLSTM Baseline | 28.0% |
| BiLSTM + FOMAML | 43.0% |
| PosCNP + MAML (original) | 50.5% |
| **PosCNP + MAML (augmented)** | **61.0%** |

## Repository Structure

```
arc_meta_learning.ipynb   # Main notebook — data loading, models, training, evaluation
report.pdf                # Full written report
presentation.pdf          # Slide deck
data/                     # 1D-ARC dataset (JSON tasks + visualisation helpers)
  dataset/                # One directory per concept class, one JSON per task
  ds_visualize/           # Dataset visualisation utilities
  All Models/             # Archived model checkpoints from intermediate experiments
models/                   # Final trained model weights
  sc/                     # Same-concept models
    bilstm_orig.pt
    bilstm_aug.pt
    fomaml_orig.pt
    fomaml_aug.pt
    poscnp_orig.pt
    poscnp_aug.pt
  cc/                     # Cross-concept models
    bilstm_orig.pt
    bilstm_aug.pt
    fomaml_orig.pt
    fomaml_aug.pt
    poscnp_orig.pt
    poscnp_aug.pt
```

## Running

The notebook is self-contained. Open `arc_meta_learning.ipynb` in Jupyter or on Kaggle (GPU recommended for MAML training). The dataset is in `data/dataset/` and model weights in `models/`.

To load a saved model and evaluate:
```python
# Example — load PosCNP same-concept (augmented) model
model.load_state_dict(torch.load("models/sc/poscnp_aug.pt"))
```

## Dependencies

```
torch torchvision numpy
```
