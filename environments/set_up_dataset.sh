#!/bin/bash

CURRENT_DIR=$(pwd)

if [[ ! "$CURRENT_DIR" == *"characters-and-dialouges-association-in-comics"* ]]; then
    echo "Error: Not in characters-and-dialouges-association-in-comics directory"
    exit 1
fi

export HF_TOKEN=$(python environments/get_hf_token.py)

hf download hal-utokyo/Manga109 --repo-type dataset --include Manga109_released_2023_12_07.zip --local-dir ./data --token $HF_TOKEN
unzip ./data/Manga109_released_2023_12_07.zip -d ./data
rm ./data/Manga109_released_2023_12_07.zip
rm -rf ./data/.cache