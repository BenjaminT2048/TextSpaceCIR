import torch.nn.functional as F

from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
# Load spaCy model


def preprocess(doc):
    # Process the text
    # doc = nlp(text)
    # Extract lemmas, filter out stopwords and punctuation
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return lemmas

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def into_query_template(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def compute_semantic_scores(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, queries: list[str], embeddings: torch.Tensor) -> torch.Tensor:
    queries_embeddings = compute_embeddings([into_query_template("Given an image description, find similar description that conforms to the caption and contains shared concept.", i) for i in queries], tokenizer, model)
    semantic_scores = (queries_embeddings @ embeddings.T)
    return semantic_scores

def find_top_k_similar_indices_ldre(tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                               embeddings: list[torch.Tensor], queries: list[list[str]],captions: list[list[str]], k: int, temp:float) -> list[int]:
    scores = []
    #semantic_scores_relative_contents = compute_embeddings([into_query_template("Given given ", i) for i in queries], tokenizer, model)
    for diversified_queries, diversified_captions in zip(queries, captions):
        semantic_scores_queries = compute_semantic_scores(tokenizer, model, diversified_queries,embeddings)
        semantic_scores_captions = compute_semantic_scores(tokenizer, model, diversified_captions, embeddings)
        weights = torch.softmax(torch.mean(torch.nn.functional.relu(semantic_scores_captions-semantic_scores_queries), dim=-1)*1/temp, dim=-1)
        semantic_score = weights.unsqueeze(0) @ semantic_scores_queries
        #scores = torch.topk(weights, k=k, dim=-1).indices
        scores.append(semantic_score)
    scores = torch.cat(scores, dim=0)
    return torch.topk(scores, k=k, dim=-1).indices, scores

def normalize_scores(scores: torch.Tensor, max_q_d_similarity: float, min_theoretical_score: float) -> torch.Tensor:
    return (scores - min_theoretical_score) / (max_q_d_similarity - min_theoretical_score)

def get_bm25_api(corpuses: list[str],
                        k1: float = 1.5, b: float = 0):
    return None


def compute_embeddings(input_texts: list[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, pooling_strategy: str = "average") -> Tensor:
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, padding=True,truncation=True, return_tensors='pt')
    batch_dict = batch_dict.to(model.device)
    outputs = model(**batch_dict)
    if pooling_strategy == "average":
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    elif pooling_strategy == "last_token":
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    else:
        raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings