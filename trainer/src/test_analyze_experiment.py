from analyze_experiment import load_training_metadata


def test_load_training_metadata():
    experiment_id = "18f10b48b35e33553b6c9535ed174eb5496a3e8a"
    data = load_training_metadata(experiment_id)
    assert data is None
