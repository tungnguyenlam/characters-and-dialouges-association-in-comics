#!/bin/bash

# if [[ "$(pwd)" == *"/characters-and-dialouges-association-in-comics" ]] || [[ "$(pwd)" == *"/workspace" ]]; then
#     echo "You are in the characters-and-dialouges-association-in-comics directory or /workspace"

#     if [ ! -d ./data/Manga109_re_2023_12_07 ]; then
#         gdown --fuzzy "https://drive.google.com/file/d/1ZvGD7g_7l9RwxnVV2KEz61BdXkzTDyBT/view?usp=sharing" -O ./data/Manga109_zipped.zip 
#         unzip -o ./data/Manga109_zipped.zip -d ./data/
#         rm ./data/Manga109_zipped.zip
#     else
#         echo "./data/Manga109_re_2023_12_07 already exists, skipping download."
#     fi


# else
#     echo "You are NOT in the characters-and-dialouges-association-in-comics directory or /workspace"
#     echo "Change the directory to characters-and-dialouges-association-in-comics in local or /workspace in docker"
# fi

CURRENT_DIR=$(pwd)

if [[ ! "$CURRENT_DIR" == *"characters-and-dialouges-association-in-comics"* ]]; then
    echo "Error: Not in characters-and-dialouges-association-in-comics directory"
    exit 1
fi

CHARACTER_DIALOGUES_DIR=""
TEMP_DIR="$CURRENT_DIR"

while [[ "$TEMP_DIR" != "/" ]]; do
    if [[ $(basename "$TEMP_DIR") == "characters-and-dialouges-association-in-comics" ]]; then
        CHARACTER_DIALOGUES_DIR="$TEMP_DIR"
        break
    fi
    TEMP_DIR=$(dirname "$TEMP_DIR")
done

ENV_FILE="$CHARACTER_DIALOGUES_DIR/../.env"

if [[ -f "$ENV_FILE" ]]; then
    echo ".env file already exists at: $ENV_FILE"
    exit 0
fi

cat > "$ENV_FILE" << 'EOF'
HF_TOKEN=
EOF

echo ".env created at: $ENV_FILE"