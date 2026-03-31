# Project Overview

This document is a short guide for explaining the project in a portfolio review, interview, or demo.

## One-Sentence Pitch

I built a Python machine-learning pipeline that classifies stellar variability from TESS light curves using a curated local dataset, engineered time-series features, baseline classifiers, and a Streamlit interface for inspecting predictions.

## Problem

Astronomy data is often messy, large, and difficult to demo. I wanted a project that used real mission data but stayed small enough to finish well.

## Approach

I split the project into two paths:

- a synthetic path for fast testing and development
- a real-data path using official TESS light curves downloaded from MAST

That let me build the software architecture first, then layer real data onto the same pipeline.

## Technical Highlights

- Curated light curves are normalized and validated before training.
- Feature extraction is lightweight and interpretable rather than black-box first.
- The training script compares multiple baseline models and saves the best artifact.
- The app can inspect curated light curves and show probabilities, features, and dataset metadata.

## Why The Dataset Is Small

The current official sample is intentionally tiny. That keeps the project finishable and easy to explain, while still demonstrating that the workflow works on real mission data.

The perfect metrics in the current run should be treated as a smoke test, not a scientific result.

## What I Would Improve Next

1. Expand the curated TESS subset significantly.
2. Add exploratory visualizations and confusion matrices.
3. Compare baseline models with a small 1D CNN.
4. Add stronger evaluation and cross-validation once the dataset grows.

## Good Talking Points

- Why I started with engineered features instead of deep learning
- How I separated local dataset management from model training
- Why manifest-driven curation makes the project easier to maintain
- How I balanced scientific credibility with achievable scope
