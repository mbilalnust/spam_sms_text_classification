from spam_sms import pipeline


def test_train_returns_metrics():
    metrics = pipeline.train(test_size=0.3, seed=3)
    assert set(metrics.keys()) == {"precision", "recall", "f1"}
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
