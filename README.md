# Characters and Dialouges Association in Comics

## Folder structure

```
characters-and-dialouges-association-in-comics
├── data
│ ├── DATA.md
│ ├── Manga109
│ └── Manga109_Dialouge
├── Dockerfile
├── environments
│ ├── py11.yml
│ └── set_up.sh
└── README.md
```

## Guide to set-up

- Make sure you are in **.../characters-and-dialouges-association-in-comics/** (the folder containing this README).

- Set up the dataset (downloads and unzips if not present):

```bash
chmod +x ./environments/set_up.sh
./environments/set_up.sh
```

Windows is not currently supported. Please download the dataset manually and organize/rename the files to match the folder structure shown above.

### For docker on Linux (recommended)

- Build the image (Use whatever image name you want, here we use comical ^\_^)

```bash
docker buildx build -t comical .
```

- Rebuild the image everytime there is an update for the environments (Ex: Add a conda package in environments/\*.yml)

- Run the image and mount the current repo to /workspace/ in the docker container, changes in /workspace will get reflected outside

```bash
docker run -it -p 8080:8080 -v "$(pwd):/workspace" comical
```

- Open `http://localhost:8080` to access jupyterlab hosted on the docker container

### For running on local (MacOS, Linux, NO Windows)

- If you use conda

```bash
conda env create -f ./environments/py11.yml
conda activate py11
```

- If you dont use conda, read the yml files and install each package by hand, since pip install is not supported
