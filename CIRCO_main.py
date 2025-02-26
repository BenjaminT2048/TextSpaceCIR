import os
import torch
import tqdm
from transformers import AutoTokenizer, AutoModel
import json
from text_retriever import compute_embeddings, find_top_k_similar_indices

def recall_k(predictions: list[list[str]], ground_truths: list[list[str]], k: int):
    hits = 0
    total_ground_truths = 0
    assert len(predictions)==len(ground_truths), f"{len(predictions)} is not equal to {len(ground_truths)}"
    for pred, gt in zip(predictions, ground_truths):
        total_ground_truths += len(gt)
        for i in gt:
            hits += i in pred
    return hits/total_ground_truths

def mean_rank(predictions: list[list[str]], ground_truths: list[list[str]], k: int):
    total_ranks = 0
    total_ground_truths = 0
    assert len(predictions)==len(ground_truths), f"{len(predictions)} is not equal to {len(ground_truths)}"
    for pred, gt in zip(predictions, ground_truths):
        total_ground_truths += len(gt)
        for i in gt:
            try:
                rank_ = pred.index(i)+1
            except ValueError:
                total_ranks += len(pred)
            else:
                total_ranks += rank_
    return total_ranks/total_ground_truths

def map_at_k(predictions: list[list[str]], ground_truths: list[list[str]], k: int):
    """
    Computes Mean Average Precision at k.

    Args:
        predictions (list of list of str]): Each sublist contains the predicted items for one query
        ground_truths (list of list[str]): Each sublist contains the ground truth items for one query
        k (int): Number of predictions to consider

    Returns:
        float: Mean Average Precision at k
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Number of predictions and ground truths must match")

    # Store AP for each query
    aps = []
    for pred, gt in zip(predictions, ground_truths):
        # Convert ground truth to set for O(1) lookup
        gt_set = set(gt)
        # Track number of relevant items found
        num_correct = 0
        # Store precision at each position where a relevant item was found
        precisions = []
        for pos, item in enumerate(pred, 1):
            if item in gt_set:
                num_correct += 1
                precisions.append(num_correct / pos)                
        # Calculate AP
        if num_correct:
            ap = sum(precisions) / min(len(gt),k) if precisions else 0
        else:
            ap = 0.0
            
        aps.append(ap)
    # Calculate MAP
    return sum(aps) / len(aps) if aps else 0.0

def main():
    # Load tokenizer and model on GPU
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("intfloat", "multilingual-e5-large-instruct"))
    model = AutoModel.from_pretrained(os.path.join("intfloat", "multilingual-e5-large-instruct"), device_map="cuda")
    with torch.no_grad():
        # Load the database and compute embeddings from the "description" field
        database = json.load(open(os.path.join("CIRCO", "documents_pool.json")))
        database = [{"description":v, "image_id":k} for k,v in database.items()]
        db_texts = [item["description"] for item in database]
        batch_size = 256
        embeddings = []
        for i in tqdm.tqdm(range(0, len(database), batch_size)):
            embeddings.append(compute_embeddings(db_texts[i:i+batch_size], tokenizer, model))
        embeddings = torch.cat(embeddings, dim=0).to(device=model.device)
        # Load queries (each query should contain a "queries" key and a "target" key)
        queries = json.load(open(os.path.join("CIRCO", "CIRCO_query.json")))
        k = 25
        # Lists to collect raw results and evaluation inputs
        all_raw_results = []
        all_predictions = []
        all_ground_truths = []
        # Process queries in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            batch_queries = [f"Description: {q['query']}\nCaption: {q['relative_caption']}\nShared Concept: {q['shared_concept']}" for q in batch]
            batch_gt = [[str(j) for j in q["gt_img_ids"]] for q in batch]
            # Retrieve the top-k similar indices for the batch of queries
            top_k_indices = find_top_k_similar_indices(tokenizer, model, embeddings, batch_queries, k).cpu()

            # Record each query's result
            for query_item, pred_indices in zip(batch, top_k_indices):
                predictions = []
                for i in pred_indices:
                    predictions.append({"image_id":database[i]["image_id"],  "description": database[i]["description"]})
                result = {
                    "query": query_item["query"],
                    "reference_image_description": query_item["reference_image_description"],
                    "relative_caption": query_item["relative_caption"],
                    "shared_concept": query_item["shared_concept"],
                    "ground_truth": query_item["gt_img_ids"],
                    "predictions": predictions,
                }
                all_raw_results.append(result)
                all_predictions.append([i['image_id'] for i in predictions])
            # Accumulate predictions and ground truths for final evaluation
            all_ground_truths.extend(batch_gt)

        # Compute final metrics from the aggregated predictions
        final_metrics = map_at_k(all_predictions, all_ground_truths, k)
        recall_k_metrics = recall_k(all_predictions, all_ground_truths, k)
        mean_rank_metrics = mean_rank(all_predictions, all_ground_truths, k)
        # Prepare the final output
        output = {
            "raw_results": all_raw_results,
            "map@k": final_metrics,
            "recall@k":recall_k_metrics,
            "mean_rank":mean_rank_metrics
        }

        # Save the results to a JSON file
        with open(f"eval_results_map{k}.json", "w") as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()