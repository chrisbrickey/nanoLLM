"""
nanoLLM/src/inference/__init__.py

Inference flows through three layers, from process boundary to core algorithm:

scripts/generate.py (CLI entry point)
-> cli.py (flags to typed configs)
-> completion.py (prompt to completion)
-> generate.py (token-by-token sampling loop)
"""
