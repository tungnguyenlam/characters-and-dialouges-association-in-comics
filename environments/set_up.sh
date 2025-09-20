#!/bin/bash

if [[ "$(pwd)" == *"/characters-and-dialouges-association-in-comics" ]] || [[ "$(pwd)" == *"/workspace" ]]; then
    echo "You are in the characters-and-dialouges-association-in-comics directory or /workspace"

    if [ ! -d ./data/Manga109 ]; then
        gdown --fuzzy "https://drive.google.com/file/d/1xfrYRbLFK1Nzi6Yc3bKEJZ-llPBJOG2w/view?usp=sharing" -O ./data/Manga109_zipped.zip 
        unzip -o ./data/Manga109_zipped.zip -d ./data/Manga109
        rm ./data/Manga109_zipped.zip
    else
        echo "./data/Manga109 already exists, skipping download."
    fi

    if [ ! -d ./data/Manga109_Dialouge ]; then
        gdown --fuzzy "https://drive.google.com/file/d/1ON4TPEKleFJX0RyIiMlg__W3YSkN9tNc/view?usp=sharing" -O ./data/Manga109_Dialouge_zipped.zip
        unzip -o ./data/Manga109_Dialouge_zipped.zip -d ./data/Manga109_Dialouge
        rm ./data/Manga109_Dialouge_zipped.zip
    else
        echo "./data/Manga109_Dialouge already exists, skipping download."
    fi

else
    echo "You are NOT in the characters-and-dialouges-association-in-comics directory or /workspace"
    echo "Change the directory to characters-and-dialouges-association-in-comics in local or /workspace in docker"
fi