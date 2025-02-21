import torch.nn.functional as F

from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def into_query_template(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def find_top_k_similar_indices(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, 
                               embeddings: torch.Tensor, queries: list[str], k: int) -> list[int]:
    queries_embeddings = compute_embeddings([into_query_template("Given an image description, find similar description that conforms to the caption and contains shared concept.", i) for i in queries], tokenizer, model)
    scores = (queries_embeddings @ embeddings.T)
    return torch.topk(scores, k=k, dim=-1).indices

def compute_embeddings(input_texts: list[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tensor:
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, padding=True,truncation=True, return_tensors='pt')
    batch_dict = batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings