from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class ConstraintType(str, Enum):
    CONTENT = "Content Constraint"
    SITUATION = "Situation Constraint"
    STYLE = "Style Constraint"
    FORMAT = "Format Constraint"
    LENGTH = "Length Constraint"

class Constraint(BaseModel):
    id: str = Field(..., description="Unique identifier for the constraint")
    type: ConstraintType
    description: str = Field(..., description="Detailed description of the constraint")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata for evaluation (e.g. word count, format type)")

class SubTask(BaseModel):
    id: str = Field(..., description="Unique identifier for the subtask")
    instruction: str = Field(..., description="The specific instruction for this subtask")
    constraints: List[Constraint] = Field(default_factory=list, description="List of constraints for this subtask")

class Prompt(BaseModel):
    id: str = Field(..., description="Unique identifier for the prompt")
    language: str = Field(default="English", description="Language of the prompt")
    core_task_category: str = Field(..., description="The core task category (e.g., Classification, Text Generation)")
    instruction: str = Field(..., description="The main overarching instruction")
    sub_tasks: List[SubTask] = Field(default_factory=list, description="List of individual subtasks")
    reading_materials: Optional[str] = Field(None, description="Any base text or dialogue provided as context")
    
    cultural_accessibility_labels: List[str] = Field(default_factory=list, description="Labels for cultural anchors")
    density_level: str = Field(..., description="Low, Medium, or High density based on constraints")

class GeneratedOutput(BaseModel):
    prompt_id: str
    model_name: str
    output_text: str
    
class EvaluationResult(BaseModel):
    prompt_id: str
    model_name: str
    rfr_score: float = Field(..., description="Requirement Following Ratio")
    ifr_score: float = Field(..., description="Instruction Following Ratio (1.0 if all met, 0.0 otherwise)")
    constraint_results: Dict[str, bool] = Field(..., description="Mapping of constraint id to boolean success")
