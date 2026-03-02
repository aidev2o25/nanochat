# vast.ai config
[build-system]
requires = ["setuptools>=80.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["nanochat*"]
exclude = ["dev*", "runs*", "tests*"]

# Getting started
ssh -i /path/to/key.pem username@server_ip
git clone https://github.com/karpathy/nanochat
pip install .
sudo apt install tmux

# Set up debugger
create a debug config file so I can run through the debugger

# Weights and biases
Set up wandb and use it
--run="[WANDB-NAME]"

#  Stop and Resume
--model-tag=[TAG-NAME]
--resume-from-step=[CHECKPOINT-STEP,e.g.5000]