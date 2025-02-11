#!/bin/bash

# URL of the Google Drive folder
FOLDER_URL="https://drive.google.com/drive/u/2/folders/10fyu9MJr9jsSU_JSr7Q3ALUPHa-WCRXz"

# Function to check if gdown is installed
check_gdown() {
    if ! command -v gdown &> /dev/null; then
        echo "gdown could not be found. Installing gdown..."
        pip install gdown
    else
        echo "gdown is already installed."
    fi
}

# Extract the folder ID from the URL
FOLDER_ID=$(echo $FOLDER_URL | grep -o 'folders/[^/]*' | cut -d'/' -f2)

# Check and install gdown if necessary
check_gdown

# Download the folder using gdown
gdown --folder $FOLDER_ID