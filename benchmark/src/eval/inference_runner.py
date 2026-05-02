import asyncio
import json
import logging
from pathlib import Path

from src.schemas.models import Prompt, GeneratedOutput
from src.eval.openrouter_client import openrouter_chat

logger = logging.getLogger(__name__)


class InferenceRunner:
    def __init__(
        self,
        models: list[str],
        api_key: str,
        concurrency: int = 5,
        checkpoint_dir: str = "results/responses",
    ):
        self.models = models
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(concurrency)
        self.checkpoint_dir = Path(checkpoint_dir)

    def _checkpoint_path(self, model: str, prompt_id: str) -> Path:
        safe = model.replace("/", "__")
        d = self.checkpoint_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{prompt_id}.json"

    def _load(self, model: str, prompt_id: str) -> GeneratedOutput | None:
        p = self._checkpoint_path(model, prompt_id)
        if p.exists():
            return GeneratedOutput(**json.loads(p.read_text(encoding="utf-8")))
        return None

    def _save(self, model: str, output: GeneratedOutput):
        p = self._checkpoint_path(model, output.prompt_id)
        p.write_text(json.dumps(output.model_dump()), encoding="utf-8")

    async def _infer(self, model: str, prompt: Prompt) -> GeneratedOutput:
        cached = self._load(model, prompt.id)
        if cached:
            logger.debug(f"[{model}] Cache hit for {prompt.id}")
            return cached

        response = await openrouter_chat(
            messages=[{"role": "user", "content": prompt.full_prompt()}],
            model=model,
            api_key=self.api_key,
            max_tokens=2000,
            temperature=0.7,
            semaphore=self.semaphore,
        )

        output = GeneratedOutput(
            prompt_id=prompt.id,
            model_name=model,
            output_text=response,
        )
        self._save(model, output)
        return output

    async def run(self, prompts: list[Prompt]) -> dict[str, list[GeneratedOutput]]:
        results: dict[str, list[GeneratedOutput]] = {}
        for model in self.models:
            logger.info(f"Running inference: {model} on {len(prompts)} prompts...")
            tasks = [self._infer(model, p) for p in prompts]
            outputs = await asyncio.gather(*tasks)
            successful = sum(1 for o in outputs if o.output_text)
            logger.info(f"[{model}] {successful}/{len(prompts)} successful")
            results[model] = list(outputs)
        return results
