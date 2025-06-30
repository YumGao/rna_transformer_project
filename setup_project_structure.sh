#!/bin/bash

# Root project folder
#mkdir -p rna_transformer_project

#cd rna_transformer_project || exit

# Data folders
mkdir -p data/raw
mkdir -p data/processed

# Source code
mkdir -p src
touch src/__init__.py src/data_utils.py src/model.py src/train.py src/evaluate.py src/preprocess.py src/config.py


# Model definitions
mkdir -p models
touch models/transformer.py

# Utilities
mkdir -p utils
touch utils/dataset.py

# Notebooks
mkdir -p notebooks
touch notebooks/dev_colab.ipynb

# Outputs
mkdir -p outputs
touch outputs/.gitkeep  # placeholder file

# Config and docs
touch config.yaml requirements.txt README.md

# For informal exploratory scripts like loading checks, debugging, quick plots
mkdir -p dev_scripts

echo "✅ Project structure created successfully under rna_transformer_project/"
