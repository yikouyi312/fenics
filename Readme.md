# Install FEniCS on Anaconda
 - Step 1: Installing Miniconda
     - Open terminal
     - IF DEFAULT SHELL IS NOT BASH, change shell to bash by running: `chsh -s /bin/bash`, Otherwise, skip this step
     - Download the installer by either:
       - Downloading the installer directly from `https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-MacOSX-x86_64.sh` 
            and moving the downloaded file into your root directory
       - Running: `curl -sL https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-MacOSX-x86_64.sh > Miniconda3-py39_4.10.3-MacOSX-x86_64.sh`
       - Running: `wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-MacOSX-x86_64.sh`
    - Run: `bash Miniconda3-py39_4.10.3-MacOSX-x86_64.sh`
    - Follow the on screen instructions and accept everything as the default versions
    - Close and reopen terminal
    - Type the command: conda list to check that it installed correctly!
 
- Step 2: run following commands in terminal:
  - `conda create -n fenicsproject -c conda-forge fenics`
  - `source activate fenicsproject`
