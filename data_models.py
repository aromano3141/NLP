from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class ConstraintType(str, Enum):
    """Types of constraints derived from EIFBENCH taxonomies."""
    CONTENT = "CONTENT"
    SITUATION = "SITUATION"
    STYLE = "STYLE"
    FORMAT = "FORMAT"
    LENGTH = "LENGTH"

class ExecutionMode(str, Enum):
    """Execution modes for combining multiple constraints (from EIFBENCH)."""
    PARALLEL = "PARALLEL"
    SERIAL = "SERIAL"
    CONDITIONAL = "CONDITIONAL"
    NESTED = "NESTED"

class Constraint(BaseModel):
    """
    Represents a specific instruction constraint.
    Supports recursive sub-constraints for NESTED execution modes,
    and activation conditions for CONDITIONAL logic.
    """
    id: str
    type: ConstraintType
    description: str
    language: str

    # Conditional logic
    activation_condition: Optional[str] = None

    # Nested constraints
    sub_constraints: Optional[List["Constraint"]] = None

    # Length specific logic
    target_min: Optional[int] = None
    target_max: Optional[int] = None
    unit: Optional[str] = None  # e.g., "words", "characters"

# Allow self-referencing model resolution for Constraint
Constraint.update_forward_refs()

class EvaluationCriteria(BaseModel):
    """
    Semantic Anchor Rubrics (from XIFBENCH methodology).
    Maps constraint IDs to specific English yes/no questions for the judge.
    """
    rubrics: Dict[str, str] = Field(description="Dictionary mapping constraint ID to an English Yes/No evaluation question.")

class Instruction(BaseModel):
    """
    Core representation of a complex prompt instruction.
    """
    id: str
    target_language: str
    base_prompt: str
    execution_mode: ExecutionMode
    constraints: List[Constraint]
    evaluation_criteria: EvaluationCriteria

class EvaluationResult(BaseModel):
    """
    Results output by the evaluation engine.
    """
    instruction_id: str
    constraint_results: Dict[str, bool] = Field(description="Dictionary mapping constraint IDs to whether they were satisfied.")
    length_score: Optional[float] = None
    holistic_pass: bool = Field(description="True only if all constraint_results are True.")
