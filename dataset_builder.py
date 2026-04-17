
from typing import List, Dict
from data_models import Constraint, ExecutionMode, ConstraintType, Instruction, EvaluationCriteria

class DatasetBuilder:
    """
    Constructs complex compositional prompts by combining constraints.
    Enforces EIFBENCH execution modes via explicit structural markers.
    """

    @staticmethod
    def build_parallel_prompt(base_prompt: str, constraints: List[Constraint]) -> str:
        """Appends constraints logically for PARALLEL execution."""
        prompt = base_prompt + "\n\nPlease ensure your response meets the following constraints:\n"
        for i, constraint in enumerate(constraints, 1):
            prompt += f"{i}. {constraint.description}\n"
        return prompt

    @staticmethod
    def build_serial_prompt(base_prompt: str, constraints: List[Constraint]) -> str:
        """Uses explicit task markers for SERIAL execution."""
        prompt = base_prompt + "\n\nYou must perform the following tasks sequentially. Do not move to the next task until the current one is completed. Use the task markers to structure your response.\n\n"
        for i, constraint in enumerate(constraints, 1):
            prompt += f"Task {i}: {constraint.description}\n"
            prompt += f"Format your output for this task between [start_task_{i}] and [end_task_{i}] tags.\n\n"
        return prompt

    @staticmethod
    def build_conditional_prompt(base_prompt: str, constraints: List[Constraint]) -> str:
        """Builds a prompt requiring CONDITIONAL logic."""
        prompt = base_prompt + "\n\nFollow these conditional instructions:\n"
        for i, constraint in enumerate(constraints, 1):
            if constraint.activation_condition:
                prompt += f"{i}. If {constraint.activation_condition}, then: {constraint.description}\n"
            else:
                prompt += f"{i}. {constraint.description}\n"
        return prompt

    @staticmethod
    def build_nested_prompt(base_prompt: str, constraints: List[Constraint]) -> str:
        """Builds a prompt with NESTED sub-tasks and constraints."""
        prompt = base_prompt + "\n\nComplete the following nested tasks:\n"

        def _build_nested(c_list: List[Constraint], level: int = 1):
            nested_str = ""
            indent = "  " * (level - 1)
            for i, constraint in enumerate(c_list, 1):
                nested_str += f"{indent}{level}.{i} {constraint.description}\n"
                if constraint.sub_constraints:
                    nested_str += _build_nested(constraint.sub_constraints, level + 1)
            return nested_str

        prompt += _build_nested(constraints)
        return prompt

    @classmethod
    def compose_instruction(cls, instruction_id: str, target_language: str, base_prompt: str, execution_mode: ExecutionMode, constraints: List[Constraint], rubrics: Dict[str, str]) -> Instruction:
        """Assembles a full Instruction object with the compiled prompt."""

        if execution_mode == ExecutionMode.PARALLEL:
            composed_prompt = cls.build_parallel_prompt(base_prompt, constraints)
        elif execution_mode == ExecutionMode.SERIAL:
            composed_prompt = cls.build_serial_prompt(base_prompt, constraints)
        elif execution_mode == ExecutionMode.CONDITIONAL:
            composed_prompt = cls.build_conditional_prompt(base_prompt, constraints)
        elif execution_mode == ExecutionMode.NESTED:
            composed_prompt = cls.build_nested_prompt(base_prompt, constraints)
        else:
            composed_prompt = base_prompt

        return Instruction(
            id=instruction_id,
            target_language=target_language,
            base_prompt=composed_prompt,
            execution_mode=execution_mode,
            constraints=constraints,
            evaluation_criteria=EvaluationCriteria(rubrics=rubrics)
        )

def generate_mock_dataset() -> List[Instruction]:
    """Generates a sample dataset with English, Spanish, and Chinese prompts."""
    dataset = []

    # 1. English - Parallel
    c1 = Constraint(id="en_p_1", type=ConstraintType.LENGTH, description="Write exactly between 50 and 100 words.", language="English", target_min=50, target_max=100, unit="words")
    c2 = Constraint(id="en_p_2", type=ConstraintType.STYLE, description="Use a formal tone.", language="English")
    dataset.append(DatasetBuilder.compose_instruction(
        instruction_id="inst_001", target_language="English", base_prompt="Explain the theory of relativity.", execution_mode=ExecutionMode.PARALLEL, constraints=[c1, c2], rubrics={"en_p_1": "Is the response length between 50 and 100 words?", "en_p_2": "Is the tone of the response formal?"}
    ))

    # 2. Spanish - Serial
    c3 = Constraint(id="es_s_1", type=ConstraintType.CONTENT, description="Resume la historia de España en 3 oraciones.", language="Spanish")
    c4 = Constraint(id="es_s_2", type=ConstraintType.FORMAT, description="Traduce el resumen al inglés.", language="Spanish")
    dataset.append(DatasetBuilder.compose_instruction(
        instruction_id="inst_002", target_language="Spanish", base_prompt="Escribe sobre la historia de España.", execution_mode=ExecutionMode.SERIAL, constraints=[c3, c4], rubrics={"es_s_1": "Does the first task contain a 3-sentence summary of Spanish history?", "es_s_2": "Does the second task contain an English translation of the summary?"}
    ))

    # 3. Chinese - Conditional
    c5 = Constraint(id="zh_c_1", type=ConstraintType.CONTENT, description="写一首关于春天的诗。", language="Chinese")
    c6 = Constraint(id="zh_c_2", type=ConstraintType.STYLE, description="确保这首诗是押韵的。", language="Chinese", activation_condition="the poem is written in classical Chinese style")
    dataset.append(DatasetBuilder.compose_instruction(
        instruction_id="inst_003", target_language="Chinese", base_prompt="关于春天写点东西。", execution_mode=ExecutionMode.CONDITIONAL, constraints=[c5, c6], rubrics={"zh_c_1": "Is the response a poem about spring?", "zh_c_2": "If the poem is written in classical Chinese style, does it rhyme?"}
    ))

    # 4. English - Nested
    c8 = Constraint(id="en_n_1a", type=ConstraintType.FORMAT, description="Format it as a bulleted list.", language="English")
    c7 = Constraint(id="en_n_1", type=ConstraintType.CONTENT, description="List 3 benefits of exercise.", language="English", sub_constraints=[c8])
    dataset.append(DatasetBuilder.compose_instruction(
        instruction_id="inst_004", target_language="English", base_prompt="Discuss exercise.", execution_mode=ExecutionMode.NESTED, constraints=[c7], rubrics={"en_n_1": "Does the response list 3 benefits of exercise?", "en_n_1a": "Are the 3 benefits formatted as a bulleted list?"}
    ))

    return dataset

if __name__ == "__main__":
    # Test generation
    mock_data = generate_mock_dataset()
    for inst in mock_data:
        print(f"--- Instruction {inst.id} ({inst.execution_mode.value}) ---")
        print(inst.base_prompt)
        print()
