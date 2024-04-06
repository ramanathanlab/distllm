"""Response Synthesizer module for RAG4Sci."""

from __future__ import annotations

from rag.search import Retriever

from distllm.generate import LLMGenerator
from distllm.generate import PromptTemplate
from distllm.generate.prompts import IdentityPromptTemplate
from distllm.generate.prompts import IdentityPromptTemplateConfig

# TODO: Consider prompt augmentation, e.g., we could make use of other
# LLM calls to rank the retrieved context according to length constraints
# or summarize the context for shorter prompt packing in the generator.

# TODO: Figure out how to load custom model weights into vllm.


class RagGenerator:
    """RAG generator for generating responses to queries."""

    def __init__(self, retriever: Retriever, generator: LLMGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def generate(
        self,
        texts: str | list[str],
        prompt_template: PromptTemplate | None = None,
        retrieval_top_k: int = 1,
        retrieval_score_threshold: float = 0.0,
    ) -> list[str]:
        """Generate a response to a query given a context.

        Parameters
        ----------
        texts : str | list[str]
            The query or queries to generate a response for.
        prompt_template : PromptTemplate, optional
            The prompt template to use. If None, will default
            to the identity prompt template, by default None.
        retrieval_top_k : int, optional
            The number of retrievals to return, by default 1.
        retrieval_score_threshold : float, optional
            The retrieval score threshold to use. Filters out
            retrievals with scores not satisfying the threshold,
            by default keep all.
        """
        # Use the identity prompt template if none is provided
        if prompt_template is None:
            prompt_template = IdentityPromptTemplate(
                IdentityPromptTemplateConfig(),
            )
        assert prompt_template is not None

        # Retrieve the search results and query embedding
        results, _ = self.retriever.search(
            texts,
            top_k=retrieval_top_k,
            score_threshold=retrieval_score_threshold,
        )

        # Get the text that corresponds to the top indices
        contexts = [
            self.retriever.get_texts(indices)
            for indices in results.total_indices
        ]

        # Preprocess the text into prompts
        prompts = prompt_template.preprocess(texts, contexts=contexts)

        # Generate a response to the query
        response = self.generator.generate(prompts)

        # Postprocess the response
        response = prompt_template.postprocess(response)

        return response


if __name__ == '__main__':
    from argparse import ArgumentParser

    from distllm.embed import get_encoder
    from distllm.embed import get_pooler
    from distllm.generate import get_generator
    from distllm.generate import get_prompt_template
    from distllm.rag.search import FaissIndex

    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
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

    text = 'What is the capital of France?'

    responses = generator.generate(texts=text, prompt_template=prompt_template)

    print(f'Question: {text}\nAnswer: {responses}')
