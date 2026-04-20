from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(relative_path: str) -> str:
    prompt_path = PROMPTS_DIR / relative_path
    return prompt_path.read_text(encoding="utf-8")
