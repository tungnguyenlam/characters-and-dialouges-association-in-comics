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

- If you want to set up the dataset manually (downloads and unzips if not present), for docker users, those 2 command will be automatically run:

```bash
chmod +x ./environments/set_up.sh
./environments/set_up.sh
```

### For docker (recommended)

- Build the image (Use whatever image name you want, here we use comical ^\_^)

```bash
docker buildx build -t comical .
```

- Rebuild the image everytime there is an update for the environments (Ex: Add a conda package in environments/\*.yml)

- Run the image and mount the current repo to /workspace/ in the docker container, changes in /workspace will get reflected outside (on host machine)

```bash
docker run -it --rm -p 8080:8080 -v "$(pwd):/workspace" comical
```

- Open `http://localhost:8080` to access jupyterlab hosted on the docker container

### For running on local (MacOS, Linux, NO Windows)

- If you use conda

```bash
conda env create -f ./environments/py11.yml
conda activate py11
```

- If you dont use conda, read the yml files and install each package by hand, since pip install requirement.txt is not supported
