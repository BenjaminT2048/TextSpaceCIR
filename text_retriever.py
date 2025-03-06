import torch.nn.functional as F

from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from rank_bm25 import BM25Okapi

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def into_query_template(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def find_top_k_similar_indices(tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                               bm25: BM25Okapi,
                               synonyms: list[dict[str, list[str]]],
                               embeddings: torch.Tensor, queries: list[str], k: int,
                               lambda_: float = 0.25,
                               top_k_bm25: int = 100) -> list[int]:
    queries_embeddings = compute_embeddings([into_query_template("Given an image description, find similar description that conforms to the caption and contains shared concept.", i) for i in queries], tokenizer, model)
    scores = (queries_embeddings @ embeddings.T)
    bm25_scores = compute_bm25_scores(bm25, synonyms, embeddings.shape[0]).to(scores.device)
    scores = scores * lambda_ + (1 - lambda_) * torch.softmax(bm25_scores, dim=-1)
    return torch.topk(scores, k=k, dim=-1).indices
    # top_k_bm25_indices = torch.topk(bm25_scores, k=top_k_bm25, dim=-1).indices
    # Find indices that are in the top-k BM25 scores for each query
    # top_k_indices = []
    # for i in range(scores.shape[0]):
    #    # Get the scores for the current query, but only for indices in top_k_bm25_scores
    #    filtered_scores = scores[i, top_k_bm25_indices[i]]
    #    # Get the top-k indices from the filtered scores
    #    top_k_filtered = torch.topk(filtered_scores, k=min(k, len(filtered_scores)), dim=-1).indices
    #    # Map back to original indices
    #    original_indices = top_k_bm25_indices[i][top_k_filtered]
    #    top_k_indices.append(original_indices)
    # return torch.stack(top_k_indices,dim=0)
    # return top_k_bm25_indices

def get_bm25_api(corpuses: list[str],
                        k1: float = 1.5, b: float = 0.75) -> BM25Okapi:
    from rank_bm25 import BM25Okapi
    corpus_tokens = [doc.lower().split() for doc in corpuses]
    return BM25Okapi(corpus_tokens, k1=k1, b=b)

def compute_bm25_scores(bm25: BM25Okapi,
                        synonyms: list[dict[str, list[str]]],
                        corpuses_len: int) -> torch.Tensor:
    """
    Compute BM25 scores for all queries against all corpus documents.
    For each word in a query, consider its synonyms and use the max BM25 score.
    For different words in a query, average their BM25 scores.
    
    Args:
        corpuses: List of corpus document strings
        synonyms: List of dictionaries mapping words to their synonyms
        k1: BM25 parameter for term frequency saturation
        b: BM25 parameter for document length normalization
    
    Returns:
        Tensor of shape (len(queries), len(corpuses)) containing BM25 scores
    """
    # Initialize scores tensor
    scores = torch.zeros((len(synonyms), corpuses_len))
    
    # Process each query
    for q_idx, query_synonyms in enumerate(synonyms):
        # Track scores for each term in the query
        term_scores = []    
        # Process each term in the query
        for term, term_synonyms in query_synonyms.items():
            # Add the original term to its synonyms
            all_terms = set([term.lower()] + [syn.lower() for syn in term_synonyms])
            # Calculate scores for each document for this term and its synonyms
            max_term_scores = torch.zeros(corpuses_len,)
            for t in all_terms:
                # Get BM25 scores for this term
                term_query = [t]
                term_score = torch.tensor(bm25.get_scores(term_query))
                # Keep the maximum score among synonyms
                max_term_scores = torch.maximum(max_term_scores, term_score)
            term_scores.append(max_term_scores)
        # Average the scores across all terms in the query
        if term_scores:
            scores[q_idx] = sum(term_scores)/len(term_scores)
    return scores


def compute_embeddings(input_texts: list[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tensor:
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, padding=True,truncation=True, return_tensors='pt')
    batch_dict = batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings