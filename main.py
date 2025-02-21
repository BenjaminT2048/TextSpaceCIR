import torch
from transformers import AutoTokenizer, AutoModel
import json
from text_retriever import compute_embeddings, find_top_k_similar_indices

def topk_eval(predictions, ground_truths):
    """
    Evaluates top-k predictions.

    Args:
        predictions (list of list of int): Each sublist contains the indices of the top-k retrieved items for one query.
        ground_truths (list of int): Each element is the ground-truth index for the corresponding query.

    Returns:
        dict: A dictionary with aggregated metrics.
            - topk_accuracy: Fraction of queries where the ground truth appears in the top-k.
            - mrr: Mean reciprocal rank.
            - mean_rank: Average rank (if the ground truth is not found, we use k+1 as the rank).
    """
    hits = 0
    reciprocal_ranks = 0.0
    total_rank = 0.0
    k = len(predictions[0]) if predictions else 0
    assert len(predictions)==len(ground_truths), f"{len(predictions)} is not equal to {len(ground_truths)}"
    for pred, gt in zip(predictions, ground_truths):
        if gt in pred:
            hits += 1
            rank = pred.index(gt) + 1  # rank is 1-indexed
            reciprocal_ranks += 1.0 / rank
            total_rank += rank
        else:
            # If not found, we assign a rank of k+1
            total_rank += (k + 1)

    num_queries = len(predictions)
    accuracy = hits / num_queries if num_queries > 0 else 0.0
    mrr = reciprocal_ranks / num_queries if num_queries > 0 else 0.0
    mean_rank = total_rank / num_queries if num_queries > 0 else 0.0

    return {"topk_accuracy": accuracy, "mrr": mrr, "mean_rank": mean_rank}

def main():
    # Load tokenizer and model on GPU
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct', device_map="cuda")
    with torch.no_grad():
        # Load the database and compute embeddings from the "description" field
        database = json.load(open("captioned_files/dress_train100.json"))
        db_texts = [item["description"] for item in database]
        batch_size = 256
        embeddings = []
        for i in range(0, len(database), batch_size):
            embeddings.append(compute_embeddings(db_texts[i:i+batch_size], tokenizer, model))
        embeddings = torch.cat(embeddings, dim=0).to(device=model.device)

        # Load queries (each query should contain a "queries" key and a "target" key)
        queries = json.load(open("captioned_files/dress_train100_query.json"))

        k = 10

        # Lists to collect raw results and evaluation inputs
        all_raw_results = []
        all_predictions = []
        all_ground_truths = []

        # Process queries in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            batch_queries = [f"Query: {q['query']}" for q in batch]
            batch_gt = [q["target"] for q in batch]

            # Retrieve the top-k similar indices for the batch of queries
            top_k_indices = find_top_k_similar_indices(tokenizer, model, embeddings, batch_queries, k).cpu()

            # Record each query's result
            for query_item, pred_indices in zip(batch, top_k_indices):
                predictions = []
                for i in pred_indices:
                    predictions.append({"target":database[i]["target"],  "description": database[i]["description"]})
                result = {
                    "query": query_item["query"],
                    "ground_truth": query_item["target"],
                    "predictions": predictions
                }
                all_raw_results.append(result)
                all_predictions.append([i['target'] for i in predictions])
            # Accumulate predictions and ground truths for final evaluation
            all_ground_truths.extend(batch_gt)

        # Compute final metrics from the aggregated predictions
        final_metrics = topk_eval(all_predictions, all_ground_truths)

        # Prepare the final output
        output = {
            "raw_results": all_raw_results,
            "metrics": final_metrics
        }

        # Save the results to a JSON file
        with open(f"100_eval_results_top{k}.json", "w") as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()