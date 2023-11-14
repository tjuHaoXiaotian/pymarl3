#!/bin/bash
# Install PyTorch and Python Packages
# 3. Install Python dependencies
echo 'Install PyTorch and Python dependencies...'
# conda create -n pymarl python=3.8 -y
# conda activate pymarl

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y
pip install sacred numpy scipy gym==0.10.8 matplotlib seaborn \
    pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger

# pip install git+https://github.com/oxwhirl/smac.git
# Do not need install SMAC anymore. We have integrated SMAC-V1 and SMAC-V2 in pymarl3/envs.
pip install "protobuf<3.21"
pip install "pysc2>=3.0.0"
pip install "s2clientprotocol>=4.10.1.75800.0"
pip install "absl-py>=0.1.0"
