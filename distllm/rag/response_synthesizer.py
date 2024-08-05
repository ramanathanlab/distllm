"""Response Synthesizer module."""

from __future__ import annotations

from distllm.generate import LLMGenerator
from distllm.generate import PromptTemplate
from distllm.generate.prompts import IdentityPromptTemplate
from distllm.generate.prompts import IdentityPromptTemplateConfig
from distllm.rag.search import Retriever

# TODO: Consider prompt augmentation, e.g., we could make use of other
# LLM calls to rank the retrieved context according to length constraints
# or summarize the context for shorter prompt packing in the generator.

# TODO: Figure out how to load custom model weights into vllm.


class RagGenerator:
    """RAG generator for generating responses to queries."""

    def __init__(
        self,
        generator: LLMGenerator,
        retriever: Retriever | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def generate(
        self,
        texts: str | list[str],
        prompt_template: PromptTemplate | None = None,
        retrieval_top_k: int = 5,
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

        # Contexts are None unless there is a retriever (no-RAG baseline).
        contexts, scores = None, None
        if self.retriever is not None:
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

            # Get the scores that correspond to the top indices
            scores = results.total_scores

        # Preprocess the text into prompts
        prompts = prompt_template.preprocess(texts, contexts, scores)

        # Generate a response to the query
        responses = self.generator.generate(prompts)

        # Postprocess the response
        responses = prompt_template.postprocess(responses)

        # Check 1-1 correspondence between queries and responses
        assert len(texts) == len(
            responses,
        ), 'Mismatch between queries and responses.'

        return responses
