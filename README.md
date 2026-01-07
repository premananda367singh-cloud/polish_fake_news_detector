📘 Polish Fake News Detector — Frontend Integration Guide

This document describes the supported integration interface for using the Polish Fake News Detector in frontend or API-based applications.

The goal is to provide a stable, minimal surface that allows frontend development without requiring knowledge of the underlying ML implementation.

Project Structure (Integration Perspective)

From an integration standpoint, the system is organized as follows:

services/    ← Supported integration layer
models/      ← Internal implementation
utils/       ← Internal helpers


Frontend and API integrations should rely on the service layer, which provides a consistent interface and abstracts internal model details.

Core Concept

The service exposes a text-analysis interface:

Input text (+ optional metadata)
        → Analysis service
        → Prediction and confidence results


The internal choice of models and ensemble logic is handled automatically.

Public API
Service Class
from services.inference import InferenceService

service = InferenceService()


This instance can be reused across requests in a web app or API.

Analyzing a Single Article
result = service.analyze_news(
    text="Ministerstwo Zdrowia poinformowało...",
    metadata={
        "source": "gov.pl",
        "author": "Ministerstwo Zdrowia",
        "date": "2025-01-01"
    }
)


Metadata is optional but can improve credibility scoring.

Response Structure (Stable Contract)

The service returns a structured dictionary containing:

prediction label (fake / real)

confidence score (0–1)

class probabilities

optional credibility information

timing metadata

This structure is designed to remain stable even if internal models change.

Batch Analysis

Batch analysis follows the same response format per item:

results = service.analyze_batch(texts, metadata_list)

Notes for Integration

The service layer is designed to be framework-agnostic
(usable with Streamlit, Flask, FastAPI, etc.)

Model training and configuration are handled separately

Internal modules may evolve without affecting the service interface

Summary

Use the service layer for integration

Treat internal modules as implementation details

Expect stable input/output contracts

No ML-specific handling required on the frontend
