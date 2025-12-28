# (OPTION 1) Environment confguration  (IGNORE IF U ARE USING MacOS)
As BNfinder2 runs on Python 2 we have to set up the environment.

## Install pyenv
Install build dependencies:
```bash
sudo apt update
sudo apt install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  wget curl llvm \
  libncurses5-dev libncursesw5-dev \
  xz-utils tk-dev \
  libffi-dev
```

Download and install pyenv:
```bash
curl https://pyenv.run | bash
```

Add pyenv to `~/.bashrc`:
```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

Reload shell:
```bash
exec $SHELL
```

## Install Python 2.7
Install python2.7.18:
```bash
pyenv install 2.7.18
```

Set up your local python version in your project directory:
```bash
cd project/dir/name
pyenv local 2.7.18
python --version
```
Version check should return `2.7.18`. Also, a `.python-version` file should be created in your directory.

## Set up virtual environment with virtualenv (compatible with python2)
Install `virtualenv` package with `pip` (if not already installed):
```bash
pip install virtualenv
```

Create and activate the virtual environment with python2 of chosen name (here `.venv2`):
```bash
virtualenv -p python2 .venv2
source .venv2/bin/activate
```

## Install BNfinder
Inside of the virtual environment:
```bash
pip install BNfinder
```

Check if it worked:
```bash
bnf --help
```


## (OPTION 2) Environment confguration on macos using Docker

# Prerequisites

Docker Desktop for macOS:
https://www.docker.com/products/docker-desktop/

Verify Docker is installed:

```bash
docker --version

# Build image

docker build --platform=linux/amd64 -t bnfinder2 .


# Run container from image

docker run -it --rm --platform=linux/amd64 \
  -v "$PWD":/work -w /work \
  bnfinder2


# Final checks
python2.7 --version

# Should print something like: Python 2.7.18

bnf --help

# Should print options for BNfinder library
```


# How to run
Enter docker container by folowing above commands and run:
```bash
python2.7 src/create_random_network.py 
```


# Run generate_bn_trajectory_dataset.py script
This script requires python3 and its dependencies are listed in `requirements3.txt` file.
It is used to generate a random boolean network and sample a random trajectory dataset from it. For usage options run:
```bash
python scripts/generate_bn_trajectory.py --help
```
Additionally, the boolean network with all of its methods is implemented as the `BN()` class in the script, so that it can be used in other scripts, e.g. for comparison of different boolean networks.