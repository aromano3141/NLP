from typing import List, Dict

def calculate_cla(constraint_results: Dict[str, bool]) -> float:
    """
    Constraint-Level Accuracy (CLA).
    Total Constraints Met / Total Constraints.
    """
    if not constraint_results:
        return 0.0

    total = len(constraint_results)
    met = sum(1 for val in constraint_results.values() if val)
    return met / total

def calculate_ila(cla: float) -> int:
    """
    Instruction-Level Accuracy (ILA).
    1 if CLA == 1.0, else 0.
    """
    return 1 if cla == 1.0 else 0

def calculate_cdi(cla_list: List[float], ila_list: List[int]) -> float:
    """
    Compositional Degradation Index (CDI).
    Average CLA - Average ILA.
    """
    if not cla_list or not ila_list:
        return 0.0

    avg_cla = sum(cla_list) / len(cla_list)
    avg_ila = sum(ila_list) / len(ila_list)

    return avg_cla - avg_ila

def calculate_bls(output_length: int, target_min: int, target_max: int) -> float:
    """
    Bounded Length Score (BLS).
    Based on LIFEBENCH length scoring logic:
    - If L_min <= x <= L_max, score = 1.0
    - If x < L_min (Under-generation), penalty: 1 - ((L_min - x) / L_min)^2
    - If x > L_max (Over-generation), penalty: 1 - ((x - L_max) / L_max)
    Clamped to minimum of 0.0.
    """
    if target_min <= output_length <= target_max:
        return 1.0

    if output_length < target_min:
        # Under-generation heavy penalty
        # Edge case: if target_min is 0 (shouldn't happen in practice but math-safe)
        if target_min == 0:
            return 1.0
        penalty_score = 1.0 - ((target_min - output_length) / target_min) ** 2
        return max(0.0, penalty_score)

    if output_length > target_max:
        # Over-generation moderate penalty
        # Edge case: if target_max is 0
        if target_max == 0:
            return 0.0

        penalty_score = 1.0 - ((output_length - target_max) / target_max)
        return max(0.0, penalty_score)

    return 0.0
