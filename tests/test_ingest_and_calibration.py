import numpy as np

from potatofl import bundle, ingest


def test_near_duplicate_groups_are_connected_components():
    H = np.zeros((4, 8), np.uint8)
    H[2] = 255  # far from the rest
    H[3, 0] = 1  # one bit from H[0] and H[1]
    groups = ingest.components(4, ingest.near_pairs(H, 4))
    assert sorted(map(sorted, groups)) == [[0, 1, 3]]


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, c = 4000, 6
    true_logits = rng.normal(0, 2, (n, c))
    p = bundle.softmax(true_logits)
    y = np.array([rng.choice(c, p=row) for row in p])
    T = bundle.fit_temperature(true_logits * 2.0, y)  # a model twice as sharp as it should be
    assert 1.7 < T < 2.3


def test_ece_is_zero_for_confident_correct_and_large_for_confident_wrong():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert bundle.ece(p, y) < 1e-9
    assert bundle.ece(p, np.array([1, 2, 0])) > 0.9


def test_quality_gate_blocks_a_poor_model_and_passes_a_good_one(monkeypatch):
    from potatofl import config
    monkeypatch.setattr(config, "SMOKE", False)
    good = {"macro_f1": 0.80, "ece_after": 0.03}
    assert bundle.quality_gate(good) == []
    assert len(bundle.quality_gate({"macro_f1": 0.60, "ece_after": 0.03})) == 1
    assert len(bundle.quality_gate({"macro_f1": 0.60, "ece_after": 0.30})) == 2
    monkeypatch.setattr(config, "SMOKE", True)  # smoke runs are exempt
    assert bundle.quality_gate({"macro_f1": 0.0, "ece_after": 1.0}) == []
