# RADAR

Requirements: PyTorch 1.7.0, Python 3.6

This document summarizes the installation instructions and steps for developing defense against adversarial malware variants via RADAR in PyTorch. The code leverages two open-source repositories: The rlkit (https://github.com/rail-berkeley/rlkit) as the reinforcement learning infrastructure, and an extended version of malware_env (https://github.com/endgameinc/gym-malware) which is based on OpenAI's Gym RL environment. For binary executable manipulations the code uses lief libraray (https://github.com/lief-project/LIEF).


***Installation Guide***

Here are the recommended steps to create a virtual environment and install the requirements.

*virtual env

conda create -n rlkit python=3.6

*sklearn*

pip3 install scikit-learn==0.18.2

*lief*

pip3 install https://github.com/lief-project/LIEF/releases/download/0.7.0/linux_lief-0.7.0_py3.6.tar.gz

*pytorch (Having a GPU is not mandatory)*

pip3 install torch==0.4.1

*gym*

pip3 install gym==0.9.2

*upx for file compression*

chmod +wrx /rlkit/torch/gym_malware/envs/controls/UPX/upx

*PyTorch*

conda install pytorch==1.7.0 cpuonly -c pytorch

Note: Some absolute paths of files may need to be changed to your local file system.

***Data***

The data was obtained from VirusTotal and cannot be shared in a public repository. However, RADAR operates on any Windows malware files. To give your malware data as seeds to generate adversarial variants and enhance defense, simply copy your malware files into the below location:
"RADAR/gym-malware/gym_malware/envs/utils/samples". The generated adversarial samples will be stored in "RADAR/gym-malware/evaded/blackbox."

***Malware Detector Models***

Pretrained MalConv, NoNeg and Ember LightGBM malware detector models are included in the code at RADAR/gym-malware/gym_malware/envs/utils under malconv.checkpoint, nonneg.checkpoint, and gradient_boosting.pkl.

***Execution***
To conduct attack emulation, run "RADAR/rlkit/torch/sac/virel_exp_Gumbel.py." 
And to enhance adversarial robustness of a specific model run "RADAR/rlkit/hardening.py."
