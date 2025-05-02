import os
import torch
import tqdm
from transformers import AutoTokenizer, AutoModel
import json
from text_retriever import compute_embeddings, compute_semantic_scores, find_top_k_similar_indices_ldre
import argparse

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

def read_json_from_path(file_path):
    """Reads a JSON dictionary from the specified file path."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print(f"Error: JSON file '{file_path}' does not contain a dictionary.", file=sys.stderr)
    return data

def create_or_load_embeddings(model_name:str, database:dict, db_texts:list[str], tokenizer, model, dataset_name):
    batch_size = 16
    embedding_name = f"{model_name.replace('/', '_')}_embeddings_{dataset_name}.pt"
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
    return embeddings

def process_batch_results(batch, top_k_indices, fused_scores, database, indexed_database):
    """Processes a batch of query results and formats them."""
    batch_raw_results = []
    batch_predictions = []
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
        batch_raw_results.append(result)
        batch_predictions.append([p['image_id'] for p in predictions])
    return batch_raw_results, batch_predictions

def main():
    parser = argparse.ArgumentParser(description="Read a JSON config from a file path.")
    parser.add_argument("--path", required=True, help="Path to the config file.")
    args = parser.parse_args()
    config = read_json_from_path(args.path)
    print("Successfully read config:")
    print(json.dumps(config, indent=4))
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="cuda")
    database = json.load(open(config["database_path"]))
    database = [{"description":v, "image_id":k} for k,v in database.items()]
    db_texts = [item["description"].lower() for item in database]
    indexed_database = {}
    for i, item in enumerate(database):
        indexed_database[item["image_id"]] = i
    with torch.no_grad():
        embeddings = create_or_load_embeddings(model_name,database, db_texts, tokenizer, model, config["dataset"])
        # Load queries (each query should contain a "queries" key and a "target" key)
        queries = json.load(open(os.path.join(config["query_path"])))
        k = config["top_k"]
        # Lists to collect raw results and evaluation inputs
        all_raw_results = []
        all_predictions = []
        all_ground_truths = []
        # Process queries in batches
        for i in tqdm.tqdm(range(0, len(queries), config['batch_size'])):
            batch = queries[i:i+config['batch_size']]
            batch_gt = [[str(j) for j in q["gt_img_ids"]] for q in batch]
            if config["method"]["type"] == "ldre":
                batch_queries = [[f"Query: {k}\nCaption: {q['relative_caption']}\nShared Concept: {q['shared_concept']}"for k in q["query"][:config["method"]["diversified_queries"]]] for q in batch]
            batch_captions = [q["reference_image_descriptions"][:config["method"]["diversified_queries"]] for q in batch]
            # Retrieve the top-k similar indices for the batch of queries
            if config["method"]["type"] == "ldre":
                top_k_indices, fused_scores = find_top_k_similar_indices_ldre(tokenizer, model, embeddings, batch_queries, batch_captions, k, config["method"]["ensemble_temperature"])
            else:
                raise NotImplementedError
            top_k_indices = top_k_indices.cpu()
            fused_scores = fused_scores.cpu()
            # Process results for the current batch
            batch_raw_results, batch_predictions = process_batch_results(
                batch, top_k_indices, fused_scores, database, indexed_database
            )
            # Accumulate results
            all_raw_results.extend(batch_raw_results)
            all_predictions.extend(batch_predictions)
            all_ground_truths.extend(batch_gt)

        # Initialize metrics variables
        final_map_metric = None
        aps = [None] * len(all_raw_results)
        final_recall_metric = None
        recalls = [None] * len(all_raw_results)

        # Compute metrics conditionally based on config
        if "map" in config.get("metrics", []):
            final_map_metric, aps = map_at_k(all_predictions, all_ground_truths, k)
        
        if "recall" in config.get("metrics", []):
            final_recall_metric, recalls = recall_k(all_predictions, all_ground_truths, k)

        # Add metrics to raw results conditionally
        for i, raw_result in enumerate(all_raw_results):
            if aps[i] is not None:
                raw_result["map@k"] = aps[i]
            if recalls[i] is not None:
                raw_result["recall@k"] = recalls[i]
        # Prepare the final output conditionally
        output = {
            "raw_results": all_raw_results,
        }
        if final_map_metric is not None:
            output["map@k"] = final_map_metric
        if final_recall_metric is not None:
            output["recall@k"] = final_recall_metric
        os.makedirs("eval_results", exist_ok=True)
        # Save the results to a JSON file
        with open(f"eval_results/{args.path.replace('configs/', '')}", "w") as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()
