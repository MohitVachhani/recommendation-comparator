import os
import json
import math
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def find_json_files_with_key(directory, key):
    files_with_key = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'document' in data and key in data['document']:
                            files_with_key.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return files_with_key

def create_output_json(files, model):
    output_data = {
        'llmRecommendation': [],
        f'{model}Recommendation': []
    }
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'document' in data:
                    document = data['document']
                    if 'llmRecommendations' in document and f'{model}Recommendations' in document:
                        llm_recs = document['llmRecommendations']
                        model_recs = document[f'{model}Recommendations']
                        if isinstance(llm_recs, list) and isinstance(model_recs, list):
                            # Sort llm_recs based on similarityScore
                            llm_recs = sorted(llm_recs, key=lambda x: x.get('similarityScore', 0), reverse=True)
                            for llm_rec, model_rec in zip(llm_recs, model_recs):
                                output_data['llmRecommendation'].append(llm_rec.get('merchantSKU', ''))
                                output_data[f'{model}Recommendation'].append(model_rec.get('merchantSKU', ''))
                        else:
                            output_data['llmRecommendation'].append(llm_recs.get('merchantSKU', ''))
                            output_data[f'{model}Recommendation'].append(model_recs.get('merchantSKU', ''))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return output_data

def calculate_recall(llm: List[str], embedding: List[str]) -> float:
    llm_set = set(llm)
    embedding_set = set(embedding)
    matched = llm_set & embedding_set
    return len(matched) / len(llm) if llm else 0.0

def dcg(recommended: List[str], relevant_set: set) -> float:
    score = 0.0
    for i, sku in enumerate(recommended):
        if sku in relevant_set:
            score += 1 / math.log2(i + 2)  # log2(rank + 1)
    return score

def ndcg(llm: List[str], embedding: List[str]) -> float:
    relevant_set = set(llm)
    ideal_dcg = dcg(llm, relevant_set)
    actual_dcg = dcg(embedding, relevant_set)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_recommendation(llm: List[str], embedding: List[str]) -> Dict[str, Any]:
    recall = calculate_recall(llm, embedding)
    ndcg_score = ndcg(llm, embedding)
    extra_skus = [sku for sku in embedding if sku not in llm]
    
    return {
        "recall": round(recall, 3),
        "ndcg_score": round(ndcg_score, 3),
        "extra_skus_in_embedding": extra_skus
    }

if __name__ == "__main__":
    base_dir = 'dataSet'
    key_to_find = 'llmRecommendations'
    models = ['qwen', 'similar']

    # Process each folder in dataSet
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            logging.info("***** Processing folder: {} *****".format(folder_name))
            
            # Find all JSON files in this folder
            files_with_llm_recommendation = find_json_files_with_key(folder_path, key_to_find)
            
            # Dictionary to store evaluation results for each model
            model_evaluation_results = {model: [] for model in models}


            for file_path in files_with_llm_recommendation:
                logging.info("***** Processing file: {} *****".format(file_path))
                merged_output_json = {
                    'llmRecommendation': []
                }
                for model in models:
                    output_json = create_output_json([file_path], model)
                    # Use set to remove duplicates while maintaining order
                    merged_output_json['llmRecommendation'] = list(dict.fromkeys(merged_output_json['llmRecommendation'] + output_json['llmRecommendation']))
                    merged_output_json[f'{model}Recommendation'] = list(dict.fromkeys(output_json[f'{model}Recommendation']))

                
           
                # Example: evaluate for each model
                evaluation_results = {}
                for model in models:
                    if merged_output_json[f'{model}Recommendation']:
                        evaluation_result = evaluate_recommendation(
                            merged_output_json['llmRecommendation'],
                            merged_output_json[f'{model}Recommendation']
                        )
                        logging.info("***** Evaluation Result for {}: {} *****".format(model, json.dumps(evaluation_result, indent=4)))
                        evaluation_results[model] = evaluation_result
                        model_evaluation_results[model].append(evaluation_result)
                    else:
                        logging.warning(f"No recommendations found for model: {model} in file: {file_path}")

                # Read the original file
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)

                # Update the original data with evaluation results
                original_data['evaluation_results'] = evaluation_results

                # Write the updated data back to the original file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, indent=4)

                logging.info("***** Updated file: {} with evaluation results. *****".format(file_path))

            # Calculate average evaluation results for each model
            average_evaluation_results = {}
            for model in models:
                if model_evaluation_results[model]:
                    avg_recall = sum(result.get('recall', 0) for result in model_evaluation_results[model]) / len(model_evaluation_results[model])
                    avg_ndcg = sum(result.get('ndcg_score', 0) for result in model_evaluation_results[model]) / len(model_evaluation_results[model])
                    average_evaluation_results[model] = {
                        'average_recall': avg_recall,
                        'average_ndcg_score': avg_ndcg
                    }
                else:
                    logging.warning(f"No evaluation results found for model: {model}")

            # Write average evaluation results to evaluation.json in the folder
            evaluation_file_path = os.path.join(folder_path, 'evaluation.json')
            with open(evaluation_file_path, 'w', encoding='utf-8') as f:
                json.dump(average_evaluation_results, f, indent=4)

            logging.info("***** Average evaluation results written to {} *****".format(evaluation_file_path))
