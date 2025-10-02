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
rm -rf ./data/__MACOSX


if [[ "$(pwd)" == *"/characters-and-dialouges-association-in-comics" ]] || [[ "$(pwd)" == *"/workspace" ]]; then
    echo "You are in the characters-and-dialouges-association-in-comics directory or /workspace"

    if [ ! -d ./data/Human_Annotate_300 ]; then
        gdown --fuzzy "https://drive.google.com/file/d/1nekCTGInk57Oe2cZvOjO7RRkUpS2mevw/view?usp=sharing" -O ./data/Human_Annotate_300.zip 
        unzip -o ./data/Human_Annotate_300.zip -d ./data/
        rm ./data/Human_Annotate_300.zip
    else
        echo "./data/Human_Annotate_300 already exists, skipping download."
    fi

else
    echo "You are NOT in the characters-and-dialouges-association-in-comics directory or /workspace"
    echo "Change the directory to characters-and-dialouges-association-in-comics in local or /workspace in docker"
fi