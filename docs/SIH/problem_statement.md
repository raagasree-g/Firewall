# SIH Problem Statement Alignment

## Problem Title
Explainable Hallucination Detection & Reliability Verification for Large Language Models

## Challenge Overview
Generative Large Language Models (LLMs) frequently produce plausible yet factually incorrect or unsupported responses ("hallucinations"). Existing evaluation metrics (e.g., ROUGE, BLEU, or LLM-as-a-judge) lack granular factual breakdown, explainability, and mathematical rigour, often misclassifying unverified context as active hallucination.

## Core Mandate
Develop an ML-centric, explainable verification engine that breaks LLM outputs into atomic claims, retrieves ground-truth evidence, classifies claims using NLI models into `SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED`, and computes empirical reliability metrics.
