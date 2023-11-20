#!/bin/bash
set -x
sudo apt update
sudo apt install gfortran libxpm4 csh -y
python3 -m pip install -r requirements.txt

git clone --recursive https://github.com/AI4EPS/QuakeFlow.git docs/lectures/codes/QuakeFlow/
git clone --recursive https://github.com/AI4EPS/INVerse.git docs/lectures/codes/QuakeFlow/INVerse/
git clone --recursive https://github.com/zhuwq0/hashpy2.git docs/lectures/codes/QuakeFlow/hashpy2/
wget https://github.com/AI4EPS/INVerse/releases/download/inverse/sac-102.0-linux_x86_64.tar.gz -P docs/lectures/codes/QuakeFlow/INVerse/
wget https://github.com/AI4EPS/INVerse/releases/download/inverse/CPS.tar.gz -P docs/lectures/codes/QuakeFlow/INVerse/
tar -xzf docs/lectures/codes/QuakeFlow/INVerse/sac-102.0-linux_x86_64.tar.gz -C docs/lectures/codes/QuakeFlow/INVerse/
tar -xzf docs/lectures/codes/QuakeFlow/INVerse/CPS.tar.gz -C docs/lectures/codes/QuakeFlow/INVerse/
wget https://github.com/AI4EPS/INVerse/releases/download/inverse/MT-Exercises.tar -P docs/lectures/codes/
tar -xf docs/lectures/codes/MT-Exercises.tar -C docs/lectures/codes/