"""Response Synthesizer module for RAG4Sci."""

# TODO: Pack the context window with retrieved context and query.

# TODO: It could make use of other LLM calls to rank the retrieved context
# according to length constraints.

# TODO: It could also run an LLM call to summarize the retrieved context
# for shorter prompt packing in the generator.

# TODO: We could use the generator for these subcalls, or we could use a
# cheaper LLM.

# TODO: Figure out how to load custom model weights into vllm.

# Sketch:
# 1. Retrieve context and query.
# 2. Tokenize the retrieved context and query (with generator tokenizer).
# 3. Implement a simple policy (FCFS) to pack the context window using p% of
#    the each retrieved context. I.e., preserve the context and pack as much
#    as possible of the retrieved context. If we use an instruct model, we
#    need the response synthesizer to be aware of particular instruct-models.
