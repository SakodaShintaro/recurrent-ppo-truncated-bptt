# Recurrent Proximal Policy Optimization using Truncated BPTT

This repository is a fork of <https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt>.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r ./requirements.txt
pip3 install torch torchvision torchaudio
pip3 install "numpy<2.0" "tensorboard>=2.16.0"
python3 train.py --config=./configs/minigrid.yaml 2>&1 | tee logs/output_$(date +"%Y%m%d_%H%M%S").txt
```
