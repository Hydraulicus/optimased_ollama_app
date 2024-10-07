from langchain.chains import create_retrieval_chain
from cachetools import TTLCache

chain_cache = TTLCache(maxsize=1000, ttl=3600)


def cache_retrieval_chain(retriever, document_chaine):
    """Custom cache function for retrieval chain."""
    # Convert retriever settings to a string (assuming they are hashable, or use a unique identifier)
    retriever_key = f"{retriever.search_type}-{retriever.search_kwargs['k']}-{retriever.search_kwargs['score_threshold']}"

    # Check if the retrieval chain is already cached
    if retriever_key in chain_cache:
        print("Using cached retrieval chain.")
        return chain_cache[retriever_key]

    # If not cached, create the chain and store it in the cache
    chaine = create_retrieval_chain(retriever, document_chaine)
    chain_cache[retriever_key] = chaine
    print("Created and cached new retrieval chain.")
    return chaine
