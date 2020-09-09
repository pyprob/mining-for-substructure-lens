Bootstrap: docker
From: ubuntu:latest

%post
apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  python3 \
  python3-dev \
  git \
  ca-certificates \
  python3-distutils \
  curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*


# NEEDED: Disable certain interactive features when installing certain packages (e.g. specify timezone, language etc)
export DEBIAN_FRONTEND=noninteractive

export PATH="/opt/conda/bin:$PATH"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

git clone https://github.com/pyprob/mining-for-substructure-lens.git /code

# install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py && rm get-pip.py

# install dark_matter dependencies
cd code

pip install -r requirements.txt && pip install pyprob
sed -i 's/pipenv run python/python3/g' simple_test_script.bash

# make working directory

%runscript

if [ $# -eq 0 ]; then
  echo "No arguments supplied"
else
  cd code
  # parse all arguments with $@
  exec $@
fi
