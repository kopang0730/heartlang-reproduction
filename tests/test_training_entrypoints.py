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
        "checkpoints/heartlang_base/checkpoint-200.pth",
        "checkpoints/pretrain/MIMIC-IV/checkpoint-200.pth",
    ]

    for snippet in required_snippets:
        assert snippet in text


def test_notebook_runs_training_script_instead_of_inlining_pipeline():
    notebook = ROOT / "notebooks" / "run_ptbxl_linear_probe.ipynb"
    text = notebook.read_text()

    assert "scripts/run_ptbxl_linear_probe.sh" in text
    assert "bash" in text


def test_result_packaging_entrypoints_exist():
    package_script = ROOT / "scripts" / "package_results.sh"
    visualizer = ROOT / "scripts" / "visualize_ptbxl_results.py"

    package_text = package_script.read_text()
    visualizer_text = visualizer.read_text()

    assert "external/HeartLang" in package_text
    assert "results/pred" in package_text
    assert "visualize_ptbxl_results.py" in package_text
    assert "tar -czf" in package_text

    assert "roc_auc_score" in visualizer_text
    assert "average_precision_score" in visualizer_text
    assert "PTBXL_QRS_superdiagnostic" in visualizer_text
