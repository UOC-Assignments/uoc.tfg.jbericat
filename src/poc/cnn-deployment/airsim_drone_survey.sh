#!/bin/bash

# https://stackoverflow.com/questions/60303997/activating-conda-environment-from-bash-script

source ~/anaconda3/etc/profile.d/conda.sh
conda activate condapy373
python3 airsim_drone_survey.py