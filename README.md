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

On the rented GPU machine:

```bash
git clone https://github.com/kopang0730/heartlang-reproduction.git
cd heartlang-reproduction
```

Create a Python environment:

```bash
conda create -n heartlang python=3.9 -y
conda activate heartlang
```

Then clone the upstream implementation inside a third-party source directory:

```bash
mkdir -p external
git clone https://github.com/PKUDigitalHealth/HeartLang.git external/HeartLang
cd external/HeartLang
pip install -r requirements.txt
```

Download PTB-XL on the GPU machine rather than on the Mac:

```bash
mkdir -p ../../data/ptb-xl
wget -r -N -c -np -P ../../data/ptb-xl https://physionet.org/files/ptb-xl/1.0.3/
```

Download official HeartLang checkpoints from Hugging Face on the GPU machine:

```bash
pip install -U huggingface_hub
huggingface-cli download PKUDigitalHealth/HeartLang --local-dir checkpoints
```

Run jobs inside `tmux` so training survives SSH disconnects:

```bash
tmux new -s heartlang
nvidia-smi
```

The first target experiment should be PTB-XL linear probing with the official pre-trained checkpoint. After it runs successfully, add full fine-tuning, random initialization, and a ResNet1D baseline.

## Remote

GitHub repository:

https://github.com/kopang0730/heartlang-reproduction
