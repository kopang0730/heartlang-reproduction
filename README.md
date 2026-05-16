# HeartLang Reproduction

Course-paper reproduction of **Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model** on PTB-XL downstream ECG classification tasks.

## Scope

This project does not aim to reproduce full HeartLang pre-training from scratch. The practical reproduction target is:

1. Load the official HeartLang pre-trained checkpoint.
2. Prepare PTB-XL downstream datasets.
3. Run linear probing and fine-tuning experiments.
4. Compare with random initialization and a supervised ECG baseline.

## Local Mac Workflow

Use the Mac for code editing, scripts, notes, and Git operations:

```bash
git clone https://github.com/kopang0730/heartlang-reproduction.git
cd heartlang-reproduction
```

Do not commit raw ECG data, checkpoints, logs, or intermediate experiment outputs. These paths are ignored by `.gitignore`.

## 4090 Training Workflow

On AutoDL, work under the data disk so downloaded ECG data and checkpoints do not fill the system disk:

```bash
cd /root/autodl-tmp
git clone https://github.com/kopang0730/heartlang-reproduction.git
cd heartlang-reproduction
```

For a private GitHub repository, run `gh auth login` first. If AutoDL does not have `gh`, use a GitHub Personal Access Token as shown in `COMMANDS.txt`.

Create the environment:

```bash
conda create -n heartlang python=3.9 -y
conda activate heartlang
```

Then run the training launcher:

```bash
bash scripts/run_ptbxl_linear_probe.sh
```

The script will clone the upstream HeartLang implementation, install dependencies, download PTB-XL, download the official checkpoint, preprocess PTB-XL, run linear probing, and run evaluation.

Useful overrides:

```bash
TASK=form SPLIT_RATIOS="0.01 0.1 1" bash scripts/run_ptbxl_linear_probe.sh
TASK=rhythm EPOCHS=50 RUN_PREPROCESS=0 bash scripts/run_ptbxl_linear_probe.sh
TRAINABLE=all TASK=superdiagnostic bash scripts/run_ptbxl_linear_probe.sh
```

Supported PTB-XL tasks:

- `superdiagnostic`
- `subdiagnostic`
- `form`
- `rhythm`

After the first successful preprocessing run, set these flags to reuse local data and checkpoints:

```bash
RUN_PREPROCESS=0 DOWNLOAD_PTBXL=0 DOWNLOAD_CHECKPOINTS=0 bash scripts/run_ptbxl_linear_probe.sh
```

Run jobs inside `tmux` so training survives SSH disconnects:

```bash
tmux new -s heartlang
nvidia-smi
```

The first target experiment should be PTB-XL linear probing with the official pre-trained checkpoint. After it runs successfully, add full fine-tuning, random initialization, and a ResNet1D baseline.

## Jupyter

You can launch training from Jupyter on the 4090 machine with:

```bash
jupyter lab
```

Open:

```text
notebooks/run_ptbxl_linear_probe.ipynb
```

The notebook calls `scripts/run_ptbxl_linear_probe.sh` instead of duplicating the pipeline. That keeps the command-line and notebook workflows consistent.

## Remote

GitHub repository:

https://github.com/kopang0730/heartlang-reproduction
