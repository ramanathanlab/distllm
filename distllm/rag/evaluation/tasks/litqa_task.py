"""Script for running the LitQA evaluation task."""

from __future__ import annotations

from typing import Literal

from distllm.utils import BaseConfig


class IdentityPromptTemplateConfig(BaseConfig):
    """Configuration for the IdentityPromptTemplate."""

    name: Literal['identity'] = 'identity'  # type: ignore[assignment]


class IdentityPromptTemplate:
    """Identity prompt."""

    def __init__(self, config: IdentityPromptTemplateConfig) -> None:
        """Initialize the IdentityPromptTemplate."""
        self.config = config

    def preprocess(
        self,
        text: str | list[str],
        contexts: list[list[str]] | None = None,
    ) -> list[str]:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to format.
        contexts : list[list[str]], optional
            The contexts to include for each text, by default None.

        Returns
        -------
        list[str]
            The formatted prompts.
        """
        if isinstance(text, str):
            text = [text]

        return text

    def postprocess(self, responses: list[str]) -> list[str]:
        """Postprocess the responses.

        Parameters
        ----------
        responses : list[str]
            The responses to postprocess.

        Returns
        -------
        list[str]
            The postprocessed responses.
        """
        return responses


if __name__ == '__main__':
    import json
    import subprocess
    from argparse import ArgumentParser
    from pathlib import Path

    from distllm.embed import get_encoder
    from distllm.embed import get_pooler
    from distllm.generate import get_generator
    from distllm.generate import get_prompt_template
    from distllm.rag.response_synthesizer import RagGenerator
    from distllm.rag.search import FaissIndex
    from distllm.rag.search import Retriever

    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', '-ds', type=Path, required=True)
    parser.add_argument('--download_dir', '-dw', type=Path, required=True)
    args = parser.parse_args()

    # Initialize the modules
    encoder = get_encoder({})
    pooler = get_pooler({})
    generator = get_generator({})
    prompt_template = get_prompt_template({})
    faiss_index = FaissIndex(args.dataset_dir)

    retriever = Retriever(
        encoder=encoder,
        pooler=pooler,
        faiss_index=faiss_index,
    )
    generator = RagGenerator(retriever=retriever, generator=generator)

    # read in the jsonl, get the questions.
    data_file = args.download_dir / 'litqa-v0.jsonl'
    data_url = 'https://raw.githubusercontent.com/Future-House/LitQA/main/litqa-v0.jsonl'
    if not data_file.exists():
        subprocess.run(
            f'curl -o {data_file} {data_url}'.split(),
            check=False,
        )

    # generate prompt templates with each question
    # Read the jsonl file
    lines = data_file.read_text().strip().split('\n')
    content = [json.loads(line) for line in lines]

    for c in content:
        print(c)

    # # Extract the text data
    # data = [item[self.config.text_field] for item in content]

    # # generate responses with each prompt template.

    # text = 'What is the capital of France?'

    # responses = generator.generate(
    #     texts=text, prompt_template=prompt_template
    # )

    # print(f'Question: {text}\nAnswer: {responses}')
