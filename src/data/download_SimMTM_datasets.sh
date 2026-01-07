#!/bin/bash

# Dataset download script for SimMTM
# Source: https://github.com/thuml/SimMTM?tab=readme-ov-file
# Downloads datasets from Tsinghua Cloud and organizes them into flat structure

set -e

# Check if datasets.zip already exists
if [ -f "datasets.zip" ]; then
    echo "Found existing datasets.zip, skipping download..."
else
    echo "Downloading SimMTM datasets from Tsinghua Cloud..."
    wget -O datasets.zip "https://cloud.tsinghua.edu.cn/f/a238e34ff81a42878d50/?dl=1"
fi

echo "Extracting datasets..."
unzip -o -q datasets.zip

# Move classification datasets to flat structure
if [ -d "datasets/classification/dataset" ]; then
    echo "Organizing classification datasets..."
    for dataset_dir in datasets/classification/dataset/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            echo "  Moving $dataset_name..."
            # Remove existing directory if present
            rm -rf "./$dataset_name"
            mv "$dataset_dir" "./"
        fi
    done
fi

# Move forecasting datasets to flat structure
if [ -d "datasets/forecasting/dataset" ]; then
    echo "Organizing forecasting datasets..."
    for dataset_dir in datasets/forecasting/dataset/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            echo "  Moving $dataset_name..."
            # Remove existing directory if present
            rm -rf "./$dataset_name"
            mv "$dataset_dir" "./"
        fi
    done
fi

# Clean up
echo "Cleaning up temporary files..."
rm -rf datasets
rm -rf __MACOSX

echo "Done! Datasets have been organized into individual folders."
