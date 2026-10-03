from iris_pipeline.evaluation.offline import passes_quality_gate


def test_quality_gate_requires_absolute_threshold():
    assert not passes_quality_gate(
        0.89,
        min_accuracy=0.90,
        baseline_accuracy=None,
        min_improvement=0.0,
    )


def test_quality_gate_requires_configured_champion_improvement():
    assert passes_quality_gate(
        0.96,
        min_accuracy=0.90,
        baseline_accuracy=0.95,
        min_improvement=0.01,
    )
    assert not passes_quality_gate(
        0.95,
        min_accuracy=0.90,
        baseline_accuracy=0.95,
        min_improvement=0.01,
    )
