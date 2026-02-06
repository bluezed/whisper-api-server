#!/bin/bash

# Conda environment name
CONDA_ENV_NAME="whisper-api"

# Update flag (default false)
UPDATE_ENV=false

# Check for --update argument
if [[ "$1" == "--update" ]]; then
    UPDATE_ENV=true
fi

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Conda not installed. Please install conda and try again."
    exit 1
fi

# Flag indicating if a new environment was created
NEW_ENV_CREATED=false

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Creating conda environment: $CONDA_ENV_NAME"
    conda create -n "$CONDA_ENV_NAME" python=3.12 -y
    NEW_ENV_CREATED=true
else
    echo "Conda environment '$CONDA_ENV_NAME' already exists."
fi

# Get conda path
CONDA_PATH=$(which conda)

# Check that conda path was found
if [ -z "$CONDA_PATH" ]; then
    echo "Failed to find conda path. Make sure conda is installed and added to PATH."
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV_NAME"
source $(dirname "$CONDA_PATH")/../etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

# Install dependencies on first creation or with --update flag
if [[ "$NEW_ENV_CREATED" == true || "$UPDATE_ENV" == true ]]; then
    # Install dependencies from requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt"
        pip install --no-cache-dir -r requirements.txt
    else
        echo "requirements.txt not found. Make sure it's in the same directory as this script."
        exit 1
    fi
fi

# Start server
echo "Starting server..."
python server.py --config config.json

echo "Server stopped."
