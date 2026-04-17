
from metrics_engine import calculate_bls, calculate_cla, calculate_ila, calculate_cdi

def test_calculate_cla():
    assert calculate_cla({}) == 0.0
    assert calculate_cla({"c1": True, "c2": True}) == 1.0
    assert calculate_cla({"c1": True, "c2": False}) == 0.5
    assert calculate_cla({"c1": False, "c2": False}) == 0.0

def test_calculate_ila():
    assert calculate_ila(1.0) == 1
    assert calculate_ila(0.99) == 0
    assert calculate_ila(0.5) == 0
    assert calculate_ila(0.0) == 0

def test_calculate_cdi():
    assert calculate_cdi([], []) == 0.0

    # 3 prompts:
    # 1: 1.0 (ILA=1)
    # 2: 0.5 (ILA=0)
    # 3: 0.5 (ILA=0)
    # Avg CLA = (1.0 + 0.5 + 0.5) / 3 = 0.666...
    # Avg ILA = (1 + 0 + 0) / 3 = 0.333...
    # CDI = 0.333...
    cdi = calculate_cdi([1.0, 0.5, 0.5], [1, 0, 0])
    assert abs(cdi - 0.3333333333333333) < 1e-6

def test_calculate_bls_within_range():
    # Exactly bounds
    assert calculate_bls(50, 50, 100) == 1.0
    assert calculate_bls(100, 50, 100) == 1.0
    # Inside
    assert calculate_bls(75, 50, 100) == 1.0
    # Exact target
    assert calculate_bls(50, 50, 50) == 1.0

def test_calculate_bls_under_generation():
    # Target 100, actual 50
    # Penalty: 1 - ((100 - 50)/100)^2 = 1 - (0.5)^2 = 1 - 0.25 = 0.75
    assert calculate_bls(50, 100, 200) == 0.75

    # Target 100, actual 0
    # Penalty: 1 - ((100 - 0)/100)^2 = 1 - 1^2 = 0
    assert calculate_bls(0, 100, 200) == 0.0

def test_calculate_bls_over_generation():
    # Target max 100, actual 150
    # Penalty: 1 - ((150 - 100)/100) = 1 - (50/100) = 0.5
    assert calculate_bls(150, 50, 100) == 0.5

    # Target max 100, actual 200
    # Penalty: 1 - ((200 - 100)/100) = 1 - 1 = 0
    assert calculate_bls(200, 50, 100) == 0.0

def test_calculate_bls_clamping():
    # Target max 100, actual 300
    # Penalty: 1 - ((300 - 100)/100) = 1 - 2 = -1.0 => Clamped to 0.0
    assert calculate_bls(300, 50, 100) == 0.0

def test_calculate_bls_zero_handling():
    # Protects against division by zero
    assert calculate_bls(10, 0, 0) == 0.0
    assert calculate_bls(0, 0, 10) == 1.0
