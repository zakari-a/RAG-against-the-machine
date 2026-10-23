"""Grounded answer generation with the mandatory Qwen/Qwen3-0.6B model."""
from typing import Any, List, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS = 20000
SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase using ONLY the provided "
    "sources. If the sources do not contain the answer, say you do not know. "
    "Be concise."
)


class Qwen:
    """Answer questions from retrieved sources with Qwen/Qwen3-0.6B."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the tokenizer and model on CPU (downloads on first run).

        Args:
            model_name: Hugging Face id of the model to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        self.model.eval()

    @staticmethod
    def format_sources(sources: List[Chunk]) -> str:
        """Render the sources as a numbered block within the context budget.

        Args:
            sources: Sources to render, in order.

        Returns:
            The numbered blocks joined by blank lines.
        """
        blocks: List[str] = []
        total = 0
        for i, source in enumerate(sources):
            block = f"[{i + 1}] {source.file_path}\n{source.text}"
            if total + len(block) > MAX_CONTEXT_CHARS:
                if not blocks:
                    blocks.append(block[:MAX_CONTEXT_CHARS])
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks)

    def answer(self, question: str, sources: List[MinimalSource]) -> str:
        """Generate an answer to ``question`` grounded in ``sources``.

        Args:
            question: Question to answer.
            sources: Retrieved sources used as the only context.

        Returns:
            The stripped answer text.

        Raises:
            RuntimeError: If generation fails inside transformers.
        """
        user = (
            f"Sources:\n{self.format_sources(sources)}\n\n"
            f"Question: {question}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model = cast(Any, self.model)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = cast(
            str, self.tokenizer.decode(generated, skip_special_tokens=True)
        )
        return text.strip()


if __name__ == "__main__":
    demo_source = MinimalSource(
        file_path="demo.md",
        first_character_index=0,
        last_character_index=52,
        content="vLLM serves models through an OpenAI-compatible server.",
    )
    print(Qwen().answer("How does vLLM serve models?", [demo_source]))
