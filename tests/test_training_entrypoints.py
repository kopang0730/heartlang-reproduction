from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_training_script_exists_and_has_required_steps():
    script = ROOT / "scripts" / "run_ptbxl_linear_probe.sh"
    text = script.read_text()

    required_snippets = [
        "PKUDigitalHealth/HeartLang",
        "physionet.org/files/ptb-xl/1.0.3",
        "QRSTokenizer.py",
        "run_class_finetuning.py",
        'TRAINABLE="${TRAINABLE:-linear}"',
        "--trainable",
        "checkpoints/pretrain/MIMIC-IV/checkpoint-200.pth",
    ]

    for snippet in required_snippets:
        assert snippet in text


def test_notebook_runs_training_script_instead_of_inlining_pipeline():
    notebook = ROOT / "notebooks" / "run_ptbxl_linear_probe.ipynb"
    text = notebook.read_text()

    assert "scripts/run_ptbxl_linear_probe.sh" in text
    assert "bash" in text
