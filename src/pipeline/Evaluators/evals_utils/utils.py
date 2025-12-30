import pandas as pd
import os
from typing import List, Dict, Optional
import json
from tabulate import tabulate
from datetime import datetime


def get_run_dir(base_dir: str, model_name: Optional[str] = None) -> str:
    """
    Create a unique run directory with timestamp and optional model name.
    
    Args:
        base_dir: Base directory for saving outputs.
        model_name: Optional model name to include in folder name.
        
    Returns:
        Path to the new run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name:
        # Sanitize model name for folder
        safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        run_folder = f"{timestamp}_{safe_model_name}"
    else:
        run_folder = timestamp
    
    run_dir = os.path.join(base_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_output_to_json(
    source_texts: List[str], 
    expected: List[str], 
    predicted: List[str], 
    file_dir: str,
    model_name: Optional[str] = None,
    create_run_folder: bool = True
) -> str:
    """
    Save translation outputs to JSON file.
    
    Args:
        source_texts: List of source texts.
        expected: List of expected translations.
        predicted: List of predicted translations.
        file_dir: Base directory for saving.
        model_name: Optional model name for folder naming.
        create_run_folder: If True, create a timestamped subfolder.
        
    Returns:
        Path to the run directory where files were saved.
    """
    if create_run_folder:
        run_dir = get_run_dir(file_dir, model_name)
    else:
        run_dir = file_dir
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
    
    file_path = os.path.join(run_dir, 'output.json')
    
    results_list = []
    for src, exp, pred in zip(source_texts, expected, predicted):
        results_list.append({
            'source_text': src,
            'expected': exp,
            'predicted': pred
        })
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)
    
    return run_dir


def save_metrics_to_json(
    metrics: Dict[str, float], 
    file_dir: str,
    model_name: Optional[str] = None,
    create_run_folder: bool = False
) -> str:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Dictionary of metric names and values.
        file_dir: Directory for saving (should be run_dir from save_output_to_json).
        model_name: Optional model name for folder naming.
        create_run_folder: If True, create a timestamped subfolder.
        
    Returns:
        Path to the directory where metrics were saved.
    """
    if create_run_folder:
        run_dir = get_run_dir(file_dir, model_name)
    else:
        run_dir = file_dir
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
    
    file_path = os.path.join(run_dir, 'metrics.json')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    
    return run_dir


def print_tabulate_results(metrics: Dict[str, float]):
    """Print evaluation metrics in a formatted table."""
    metrics_data = [
        ["Model Name", metrics.get('model_name', 'unknown')],
        ["Character Error Rate (CER)", f"{metrics['cer']:.4f}"],
        ["Word Error Rate (WER)", f"{metrics['wer']:.4f}"],
        ["BLEU Score", f"{metrics['bleu']:.4f}"],
        ["SacreBLEU Score", f"{metrics['sacrebleu']:.4f}"],
        ["chrF Score", f"{metrics['chrf']:.4f}"],
        ["chrF++ Score", f"{metrics['chrf_pp']:.4f}"],
    ]
    
    # Add neural metrics if available
    if 'bertscore_f1' in metrics and metrics['bertscore_f1'] is not None:
        metrics_data.extend([
            ["BERTScore F1", f"{metrics['bertscore_f1']:.4f}"],
            ["BERTScore Precision", f"{metrics['bertscore_precision']:.4f}"],
            ["BERTScore Recall", f"{metrics['bertscore_recall']:.4f}"],
        ])
    
    if 'comet' in metrics and metrics['comet'] is not None:
        metrics_data.append(["COMET Score", f"{metrics['comet']:.4f}"])
    
    print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))