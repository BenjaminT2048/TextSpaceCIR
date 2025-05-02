import os
import torch
import tqdm
from transformers import AutoTokenizer, AutoModel
import json
from text_retriever import compute_embeddings, compute_semantic_scores, find_top_k_similar_indices, get_bm25_api

def recall_k(predictions: list[list[str]], ground_truths: list[list[str]], k: int):
    hits = 0
    total_ground_truths = 0
    recalls = []
    assert len(predictions)==len(ground_truths), f"{len(predictions)} is not equal to {len(ground_truths)}"
    for pred, gt in zip(predictions, ground_truths):
        total_ground_truths += len(gt)
        current_hits = 0
        for i in gt:
            if i in pred:
                current_hits += 1
        hits += current_hits
        recalls.append([f"{current_hits}/{len(gt)}", current_hits/len(gt)])
    return hits/total_ground_truths, recalls

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
    return sum(aps) / len(aps) if aps else 0.0, aps

def main():
    # Load tokenizer and model on GPU
    #model_name = "infly/inf-retriever-v1-1.5b"
    model_name = "intfloat/multilingual-e5-large-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="cuda")
    database = json.load(open(os.path.join("CIRCO", "documents_pool_1.json")))
    database = [{"description":v, "image_id":k} for k,v in database.items()]
    db_texts = [item["description"].lower() for item in database]
    indexed_database = {}
    batch_size = 16
    for i, item in enumerate(database):
        indexed_database[int(item["image_id"])] = i
    with torch.no_grad():
        embedding_name = f"{model_name.replace('/', '_')}_embeddings.pt"
        if os.path.exists(embedding_name):
            embeddings = torch.load(embedding_name)
        else:
            # Load the database and compute embeddings from the "description" field

            embeddings = []
            for i in tqdm.tqdm(range(0, len(database), batch_size)):
                embeddings.append(compute_embeddings(db_texts[i:i+batch_size], tokenizer, model))
            embeddings = torch.cat(embeddings, dim=0)
            torch.save(embeddings, embedding_name)
        embeddings = embeddings.to(model.device)
        embeddings_list = [embeddings, torch.load(embedding_name.replace("_embeddings.pt", "_embeddings_2.pt")).to(model.device)]
        bm25 = get_bm25_api(db_texts)
        # Load queries (each query should contain a "queries" key and a "target" key)
        queries = json.load(open(os.path.join("CIRCO", "CIRCO_query_diversified_temp2.json")))
        k = 25
        # Lists to collect raw results and evaluation inputs
        all_raw_results = []
        all_predictions = []
        all_ground_truths = []
        # Process queries in batches
        for i in tqdm.tqdm(range(0, len(queries), batch_size)):
            batch = queries[i:i+batch_size]
            batch_gt = [[str(j) for j in q["gt_img_ids"]] for q in batch]
            batch_queries = [[f"Query: {k}\nCaption: {q['relative_caption']}\nShared Concept: {q['shared_concept']}"for k in q["query"]] for q in batch]
            batch_captions = [q["reference_image_descriptions"] for q in batch]
            # Retrieve the top-k similar indices for the batch of queries
            top_k_indices, fused_scores = find_top_k_similar_indices(tokenizer, model,bm25, None, embeddings_list, batch_queries, batch_captions, k, None, None)
            top_k_indices = top_k_indices.cpu()
            fused_scores = fused_scores.cpu()
            # Record each query's result
            for query_item, pred_indices, fused_score in zip(batch, top_k_indices, fused_scores):
                predictions = []
                for i in pred_indices:
                    predictions.append({"image_id":database[i]["image_id"],  "description": database[i]["description"],"fused_score": float(fused_score[i])})
                ground_truths = []
                for i in query_item["gt_img_ids"]:
                    index = indexed_database[i]
                    ground_truths.append({"image_id":i, "description": database[index]["description"], "fused_score": float(fused_score[index])})
                result = {
                    "query": query_item["query"],
                    "reference_image_description": query_item["reference_image_descriptions"],
                    "relative_caption": query_item["relative_caption"],
                    "shared_concept": query_item["shared_concept"],
                    "ground_truth": ground_truths,
                    "predictions": predictions,
                }
                all_raw_results.append(result)
                all_predictions.append([i['image_id'] for i in predictions])
            # Accumulate predictions and ground truths for final evaluation
            all_ground_truths.extend(batch_gt)

        # Compute final metrics from the aggregated predictions
        final_metrics,aps = map_at_k(all_predictions, all_ground_truths, k)
        recall_k_metrics, recalls = recall_k(all_predictions, all_ground_truths, k)
        mean_rank_metrics = mean_rank(all_predictions, all_ground_truths, k)
        for ap, raw_result, recall in zip(aps,all_raw_results, recalls):
            raw_result["map@k"] = ap
            raw_result["recall@k"] = recall
        # Prepare the final output
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