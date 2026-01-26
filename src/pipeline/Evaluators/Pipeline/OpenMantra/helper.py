import gc
import json
import os
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kendalltau
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore, SacreBLEUScore, CHRFScore
from torchmetrics.text.bert import BERTScore
from torch.utils.data import Dataset
from PIL import Image
# Import COMET if available
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
# Import pipeline classes - adjust the import path based on your project structure
# Since this is in src/pipeline/Evaluators/Pipeline/OpenMantra/helper.py
# and MangaPipeline is likely in src/pipeline/Utils/MangaPipeline.py
try:
    from src.pipeline.Utils.MangaPipeline import MangaPipeline
except ImportError:
    # Alternative import if running from different directory
    from ....Utils.MangaPipeline import MangaPipeline

def match_bubbles_to_gt_texts(
    pred_bboxes: List[List[float]], 
    gt_texts: List[Dict],
    containment_threshold: float = 0.8
) -> Tuple[Dict[int, List[int]], List[int], List[int]]:
    """
    Match GT text bboxes to predicted bubbles based on containment.
    
    Args:
        pred_bboxes: List of predicted bubble bboxes [x1, y1, x2, y2]
        gt_texts: List of GT text dicts with 'xmin', 'ymin', 'xmax', 'ymax', 'text_ja', 'text_en'
        containment_threshold: Fraction of GT text that must be inside bubble
    
    Returns:
        matches: Dict mapping pred_idx -> [list of gt_indices it contains]
        unmatched_gt: GT text indices not contained in any bubble
        empty_bubbles: Predicted bubble indices containing no GT text
    """
    matches = {i: [] for i in range(len(pred_bboxes))}
    matched_gt = set()
    
    for pred_idx, bbox in enumerate(pred_bboxes):
        bx1, by1, bx2, by2 = bbox
        
        for gt_idx, gt in enumerate(gt_texts):
            gx1, gy1 = gt['xmin'], gt['ymin']
            gx2, gy2 = gt['xmax'], gt['ymax']
            
            # Compute intersection
            ix1 = max(bx1, gx1)
            iy1 = max(by1, gy1)
            ix2 = min(bx2, gx2)
            iy2 = min(by2, gy2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                gt_area = (gx2 - gx1) * (gy2 - gy1)
                
                if gt_area > 0:
                    containment = intersection / gt_area
                    if containment >= containment_threshold:
                        matches[pred_idx].append(gt_idx)
                        matched_gt.add(gt_idx)
    
    unmatched_gt = [i for i in range(len(gt_texts)) if i not in matched_gt]
    empty_bubbles = [i for i, texts in matches.items() if len(texts) == 0]
    
    return matches, unmatched_gt, empty_bubbles


def evaluate_ocr_page(
    ocr_texts: List[str],
    gt_texts: List[Dict],
    matches: Dict[int, List[int]],
    unmatched_gt: List[int]
) -> Tuple[List[str], List[str]]:
    """
    Prepare OCR predictions and expected texts for a page.
    Concatenates GT texts for bubbles containing multiple text regions.
    Handles missed bubbles (unmatched GT) by assigning empty strings.
    
    Returns:
        (predictions, expected) lists for this page
    """
    predictions = []
    expected = []
    
    # Handle matched bubbles
    for pred_idx, gt_indices in matches.items():
        if not gt_indices:
            continue
        
        # Sort GT indices to maintain reading order, then concatenate
        sorted_gt_indices = sorted(gt_indices)
        expected_text = ''.join([gt_texts[i]['text_ja'] for i in sorted_gt_indices])
        ocr_output = ocr_texts[pred_idx] if pred_idx < len(ocr_texts) else ''
        
        predictions.append(ocr_output)
        expected.append(expected_text)
        
    # Handle unmatched GT (missed bubbles)
    for gt_idx in unmatched_gt:
        expected_text = gt_texts[gt_idx]['text_ja']
        predictions.append("")
        expected.append(expected_text)
    
    return predictions, expected


def evaluate_translation_page(
    translations: List[str],
    ocr_texts: List[str],
    gt_texts: List[Dict],
    matches: Dict[int, List[int]],
    unmatched_gt: List[int]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Prepare translation predictions, expected texts, and source texts for a page.
    Concatenates GT translations for bubbles containing multiple text regions.
    Handles missed bubbles (unmatched GT) by assigning empty strings.
    
    Returns:
        (predictions, expected, sources) lists for this page
    """
    predictions = []
    expected = []
    sources = []
    
    # Handle matched bubbles
    for pred_idx, gt_indices in matches.items():
        if not gt_indices:
            continue
        
        sorted_gt_indices = sorted(gt_indices)
        expected_trans = ' '.join([gt_texts[i]['text_en'] for i in sorted_gt_indices])
        pred_trans = translations[pred_idx] if pred_idx < len(translations) else ''
        source_text = ocr_texts[pred_idx] if pred_idx < len(ocr_texts) else ''
        
        predictions.append(pred_trans)
        expected.append(expected_trans)
        sources.append(source_text)
        
    # Handle unmatched GT (missed bubbles)
    for gt_idx in unmatched_gt:
        expected_trans = gt_texts[gt_idx]['text_en']
        predictions.append("")
        expected.append(expected_trans)
        sources.append("") # No OCR source for missed bubble
    
    return predictions, expected, sources

def evaluate_ordering_page(
    matches: Dict[int, List[int]],
    num_gt_texts: int
) -> Dict[str, Any]:
    """
    Evaluate reading order for a page.
    
    Compares the order of predicted bubbles (by their index) 
    against GT reading order (GT indices are already in reading order).
    
    Returns:
        Dict with kendall_tau, exact_match, num_matched_bubbles
    """
    # Build mapping: for bubbles with GT, what's the minimum GT index they contain?
    # This represents the "reading position" of each bubble
    bubble_first_gt = {}
    for pred_idx, gt_indices in matches.items():
        if gt_indices:
            bubble_first_gt[pred_idx] = min(gt_indices)
    
    if len(bubble_first_gt) < 2:
        return {
            'kendall_tau': 1.0,
            'exact_match': True,
            'num_matched_bubbles': len(bubble_first_gt)
        }
    
    # Get prediction order (pred indices in their natural order)
    pred_order = sorted(bubble_first_gt.keys())
    
    # Get the GT ranks for each predicted bubble (in prediction order)
    gt_ranks = [bubble_first_gt[p] for p in pred_order]
    
    # Expected: gt_ranks should be monotonically increasing
    # Compute Kendall's Tau: correlation between [0,1,2,...] and gt_ranks
    ideal_sequence = list(range(len(gt_ranks)))
    tau, _ = kendalltau(ideal_sequence, gt_ranks)
    
    # Handle NaN (can happen if all values are identical)
    if np.isnan(tau):
        tau = 1.0
    
    # Check exact match (is gt_ranks sorted?)
    is_sorted = all(gt_ranks[i] <= gt_ranks[i+1] for i in range(len(gt_ranks)-1))
    
    return {
        'kendall_tau': tau,
        'exact_match': is_sorted,
        'num_matched_bubbles': len(bubble_first_gt)
    }


def evaluate_page(
    pipeline: MangaPipeline,
    image: np.ndarray,
    gt_texts: List[Dict],
    containment_threshold: float = 0.8,
    conf_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate pipeline on a single page.
    
    Returns dict with:
        - ocr_predictions, ocr_expected: Lists for OCR evaluation
        - trans_predictions, trans_expected: Lists for translation evaluation
        - ordering: Dict with ordering metrics
        - coverage: Dict with coverage metrics
    """
    # Run pipeline
    _, results = pipeline.process(image, conf_threshold=conf_threshold, return_intermediate=True)
    
    pred_bboxes = results.get('bboxes', [])
    ocr_texts = results.get('ocr_texts', [])
    translations = results.get('translated_texts', [])
    
    # Filter out empty GT texts
    valid_gt_texts = [gt for gt in gt_texts if gt.get('text_ja', '').strip() and gt.get('text_en', '').strip()]
    
    if len(pred_bboxes) == 0 or len(valid_gt_texts) == 0:
        return {
            'ocr_predictions': [],
            'ocr_expected': [],
            'trans_predictions': [],
            'trans_expected': [],
            'trans_sources': [],
            'ordering': {'kendall_tau': 0.0, 'exact_match': False, 'num_matched_bubbles': 0},
            'coverage': {
                'gt_coverage': 0.0,
                'bubble_utilization': 0.0,
                'num_gt_texts': len(valid_gt_texts),
                'num_pred_bubbles': len(pred_bboxes),
                'num_matched_gt': 0,
                'num_unmatched_gt': len(valid_gt_texts),
                'num_empty_bubbles': len(pred_bboxes)
            },
            'pred_bboxes': pred_bboxes,
            'valid_gt_texts': valid_gt_texts,
            'matches': {},
            'trans_predictions_gt_source': [],
            'trans_expected_gt_source': [],
            'trans_sources_gt_source': []
        }
    
    # Match bubbles to GT texts
    matches, unmatched_gt, empty_bubbles = match_bubbles_to_gt_texts(
        pred_bboxes, valid_gt_texts, containment_threshold
    )
    
    # --- Experiment: Translate using GT Source Text (Simulate Perfect OCR) ---
    gt_source_texts = []
    for pred_idx in range(len(pred_bboxes)):
        gt_indices = matches.get(pred_idx, [])
        if gt_indices:
            # If matched, use the concatenation of GT Japanese text
            sorted_indices = sorted(gt_indices)
            gt_text_ja = ''.join([valid_gt_texts[i]['text_ja'] for i in sorted_indices])
            gt_source_texts.append(gt_text_ja)
        else:
            # If false positive (no GT), keep the OCR text (system still sees this text)
            gt_source_texts.append(ocr_texts[pred_idx] if pred_idx < len(ocr_texts) else "")
            
    # Run translator on the "Perfect OCR" input
    # Note: We assume 'image_info' is not strictly needed for context if using text list,
    # but the translator might need it. We will pass basic info.
    # To keep it simple and consistent with pipeline.process, we use the translator directly.
    # We need to handle the case where pipeline.translator might expect image context,
    # but ContextAwareLLMTranslator mainly uses the text list.
    
    # We need to construct speakers list if possible, but for now we'll pass None
    # to let the translator handle it (it treats them as "unknown").
    
    try:
        if hasattr(pipeline.translator, 'translate_page'):
             trans_preds_from_gt = pipeline.translator.translate_page(gt_source_texts)
        elif hasattr(pipeline.translator, 'predict'): # Fallback for simple translators
             trans_preds_from_gt = pipeline.translator.predict(gt_source_texts)
        else:
             trans_preds_from_gt = [""] * len(gt_source_texts)
    except Exception as e:
        print(f"Warning: GT Source translation failed: {e}")
        trans_preds_from_gt = [""] * len(gt_source_texts)

    # Evaluate GT Source Translation
    # We reuse evaluate_translation_page but pass the new predictions and new source
    trans_preds_gt, trans_expected_gt, trans_sources_gt = evaluate_translation_page(
        trans_preds_from_gt, gt_source_texts, valid_gt_texts, matches, unmatched_gt
    )
    
    # Evaluate OCR
    ocr_preds, ocr_expected = evaluate_ocr_page(ocr_texts, valid_gt_texts, matches, unmatched_gt)
    
    # Evaluate Translation
    trans_preds, trans_expected, trans_sources = evaluate_translation_page(
        translations, ocr_texts, valid_gt_texts, matches, unmatched_gt
    )
    
    # Evaluate Ordering
    ordering = evaluate_ordering_page(matches, len(valid_gt_texts))
    
    # Coverage metrics
    num_matched_gt = len(valid_gt_texts) - len(unmatched_gt)
    coverage = {
        'gt_coverage': num_matched_gt / len(valid_gt_texts) if valid_gt_texts else 1.0,
        'bubble_utilization': (len(pred_bboxes) - len(empty_bubbles)) / len(pred_bboxes) if pred_bboxes else 1.0,
        'num_gt_texts': len(valid_gt_texts),
        'num_pred_bubbles': len(pred_bboxes),
        'num_matched_gt': num_matched_gt,
        'num_unmatched_gt': len(unmatched_gt),
        'num_empty_bubbles': len(empty_bubbles)
    }
    
    return {
        'ocr_predictions': ocr_preds,
        'ocr_expected': ocr_expected,
        'trans_predictions': trans_preds,
        'trans_expected': trans_expected,
        'trans_predictions_gt_source': trans_preds_gt,
        'trans_expected_gt_source': trans_expected_gt, # Should be identical to trans_expected
        'trans_sources_gt_source': trans_sources_gt,
        'trans_sources': trans_sources,
        'ordering': ordering,
        'coverage': coverage,
        'pred_bboxes': pred_bboxes,
        'valid_gt_texts': valid_gt_texts,
        'matches': matches
    }


def visualize_matching(
    image: np.ndarray,
    pred_bboxes: List[List[float]],
    gt_texts: List[Dict],
    matches: Dict[int, List[int]],
    save_path: Optional[str] = None
):
    """Visualize GT vs Predicted bboxes with matching."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    # Plot 1: GT text bboxes
    axes[0].imshow(image)
    axes[0].set_title("GT Text Bboxes", fontsize=12)
    for i, gt in enumerate(gt_texts):
        rect = plt.Rectangle(
            (gt['xmin'], gt['ymin']), 
            gt['xmax'] - gt['xmin'], 
            gt['ymax'] - gt['ymin'],
            fill=False, edgecolor='green', linewidth=3
        )
        axes[0].add_patch(rect)
        axes[0].text(gt['xmin'], gt['ymin'] - 5, f"GT{i}", fontsize=10, color='green')
    axes[0].axis('off')
    
    # Plot 2: Predicted bubbles with order
    axes[1].imshow(image)
    axes[1].set_title("Predicted Bubbles (Ordered)", fontsize=12)
    for i, bbox in enumerate(pred_bboxes):
        rect = plt.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            fill=False, edgecolor='red', linewidth=3
        )
        axes[1].add_patch(rect)
        axes[1].text(bbox[0], bbox[1] - 5, f"[{i}]", fontsize=10, color='red', fontweight='bold')
    axes[1].axis('off')
    
    # Plot 3: Matching overlay
    axes[2].imshow(image)
    axes[2].set_title("Matching (Green=GT, Colors=Pred)", fontsize=12)
    for i, gt in enumerate(gt_texts):
        rect = plt.Rectangle(
            (gt['xmin'], gt['ymin']), 
            gt['xmax'] - gt['xmin'], 
            gt['ymax'] - gt['ymin'],
            fill=False, edgecolor='green', linewidth=3, linestyle='--'
        )
        axes[2].add_patch(rect)
    for pred_idx, gt_indices in matches.items():
        if pred_idx < len(pred_bboxes):
            bbox = pred_bboxes[pred_idx]
            color = 'blue' if gt_indices else 'red'
            rect = plt.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                fill=False, edgecolor=color, linewidth=3
            )
            axes[2].add_patch(rect)
            label = f"P{pred_idx}→GT{gt_indices}" if gt_indices else f"P{pred_idx}(empty)"
            axes[2].text(bbox[0], bbox[1] - 5, label, fontsize=9, color=color)
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_translation_output(
    ocr_preds: List[str],
    ocr_expected: List[str],
    trans_preds: List[str],
    trans_expected: List[str],
    trans_sources: List[str],
    image_info: Dict,
    save_path: str
):
    """Save OCR and translation outputs for a page."""
    output = {
        'image_info': image_info,
        'results': []
    }
    
    max_len = max(len(ocr_preds), len(trans_preds), len(trans_sources))
    for i in range(max_len):
        output['results'].append({
            'bubble_idx': i,
            'ocr_predicted': ocr_preds[i] if i < len(ocr_preds) else '',
            'ocr_expected': ocr_expected[i] if i < len(ocr_expected) else '',
            'translation_source': trans_sources[i] if i < len(trans_sources) else '',
            'translation_predicted': trans_preds[i] if i < len(trans_preds) else '',
            'translation_expected': trans_expected[i] if i < len(trans_expected) else ''
        })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def create_summary_plots(all_results: List[Dict], save_dir: str):
    """Create summary plots for the evaluation."""
    # Plot 1: Coverage distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    coverages = [r['coverage']['gt_coverage'] for r in all_results]
    axes[0, 0].hist(coverages, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(coverages), color='red', linestyle='--', label=f'Mean: {np.mean(coverages):.3f}')
    axes[0, 0].set_xlabel('GT Coverage')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('GT Coverage Distribution')
    axes[0, 0].legend()
    
    # Plot 2: Kendall's Tau distribution
    taus = [r['ordering']['kendall_tau'] for r in all_results if r['ordering']['num_matched_bubbles'] >= 2]
    axes[0, 1].hist(taus, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(np.mean(taus), color='red', linestyle='--', label=f'Mean: {np.mean(taus):.3f}')
    axes[0, 1].set_xlabel("Kendall's Tau")
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Ordering Accuracy Distribution')
    axes[0, 1].legend()
    
    # Plot 3: Bubble counts (colored by page index)
    gt_counts = [r['coverage']['num_gt_texts'] for r in all_results]
    pred_counts = [r['coverage']['num_pred_bubbles'] for r in all_results]
    page_indices = list(range(len(all_results)))
    
    scatter = axes[1, 0].scatter(gt_counts, pred_counts, c=page_indices, cmap='viridis', alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[1, 0].plot([0, max(gt_counts)], [0, max(gt_counts)], 'r--', label='Perfect match')
    axes[1, 0].set_xlabel('GT Text Count')
    axes[1, 0].set_ylabel('Predicted Bubble Count')
    axes[1, 0].set_title('GT vs Predicted Counts')
    axes[1, 0].legend()
    
    # Add colorbar to show page index
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('Page Index')
    
    # Plot 4: Per-book metrics
    from collections import defaultdict
    book_metrics = defaultdict(list)
    for r in all_results:
        book = r['image_info']['book_title']
        book_metrics[book].append(r['coverage']['gt_coverage'])
    
    books = list(book_metrics.keys())
    means = [np.mean(book_metrics[b]) for b in books]
    axes[1, 1].barh(books, means, color='steelblue')
    axes[1, 1].set_xlabel('Mean GT Coverage')
    axes[1, 1].set_title('Coverage by Book')
    axes[1, 1].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plots saved to {save_dir}/summary_plots.png")


def compute_final_metrics(all_results: List[Dict], device: str = 'cpu') -> Dict[str, float]:
    """
    Aggregate results from all pages into final metrics.
    """
    # Resolve 'auto' device to actual device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Collect all predictions/expected across pages
    all_ocr_pred = []
    all_ocr_expected = []
    all_trans_pred = []
    all_trans_expected = []
    all_trans_sources = []
    
    # GT Source Experiment
    all_trans_pred_gt = []
    all_trans_expected_gt = []
    all_trans_sources_gt = []
    
    ordering_taus = []
    ordering_exact = []
    coverages = []
    utilizations = []
    
    # Detection statistics
    total_gt_texts = 0
    total_matched_pairs = 0
    total_missed_detections = 0
    total_false_positives = 0
    
    for result in all_results:
        all_ocr_pred.extend(result['ocr_predictions'])
        all_ocr_expected.extend(result['ocr_expected'])
        all_trans_pred.extend(result['trans_predictions'])
        all_trans_expected.extend(result['trans_expected'])
        all_trans_sources.extend(result.get('trans_sources', []))
        
        # Accumulate GT Source results
        all_trans_pred_gt.extend(result.get('trans_predictions_gt_source', []))
        all_trans_expected_gt.extend(result.get('trans_expected_gt_source', []))
        all_trans_sources_gt.extend(result.get('trans_sources_gt_source', []))
        
        if result['ordering']['num_matched_bubbles'] >= 2:
            ordering_taus.append(result['ordering']['kendall_tau'])
            ordering_exact.append(result['ordering']['exact_match'])
        
        coverages.append(result['coverage']['gt_coverage'])
        utilizations.append(result['coverage']['bubble_utilization'])
        
        # Accumulate detection statistics
        coverage = result['coverage']
        total_gt_texts += coverage['num_gt_texts']
        total_matched_pairs += coverage['num_matched_gt']
        total_missed_detections += coverage['num_unmatched_gt']
        total_false_positives += coverage['num_empty_bubbles']
    
    metrics = {}
    
    # OCR Metrics
    if all_ocr_pred and all_ocr_expected:
        cer_metric = CharErrorRate()
        metrics['ocr_cer'] = cer_metric(all_ocr_pred, all_ocr_expected).item()
    else:
        metrics['ocr_cer'] = 1.0
    
    # Translation Metrics
    if all_trans_pred and all_trans_expected:
        cer_metric = CharErrorRate()
        wer_metric = WordErrorRate()
        bleu_metric = BLEUScore()
        sacrebleu_metric = SacreBLEUScore()
        chrf_metric = CHRFScore(n_char_order=6, n_word_order=0)  # chrF
        chrf_pp_metric = CHRFScore(n_char_order=6, n_word_order=2)  # chrF++
        
        bleu_refs = [[ref] for ref in all_trans_expected]
        
        metrics['trans_cer'] = cer_metric(all_trans_pred, all_trans_expected).item()
        metrics['trans_wer'] = wer_metric(all_trans_pred, all_trans_expected).item()
        metrics['trans_bleu'] = bleu_metric(all_trans_pred, bleu_refs).item()
        metrics['trans_sacrebleu'] = sacrebleu_metric(all_trans_pred, bleu_refs).item()
        metrics['trans_chrf'] = chrf_metric(all_trans_pred, bleu_refs).item()
        metrics['trans_chrf_pp'] = chrf_pp_metric(all_trans_pred, bleu_refs).item()
        
        # Cleanup string metrics
        del cer_metric, wer_metric, bleu_metric, sacrebleu_metric, chrf_metric, chrf_pp_metric
        gc.collect()

        # BERTScore - with improved error handling
        try:
            print("Computing BERTScore...")
            print(f"  Number of samples: {len(all_trans_pred)}")
            
            # Use CPU for BERTScore if MPS causes issues
            bert_device = 'cpu' if device == 'mps' else device
            print(f"  BERTScore device: {bert_device}")
            
            bert_scorer = BERTScore(
                model_name_or_path="microsoft/deberta-xlarge-mnli",
                device=bert_device
            )
            bert_scorer.reset()
            # Pass all predictions/expected, including empty strings
            bert_scorer.update(all_trans_pred, all_trans_expected)
            bert_results = bert_scorer.compute()
            
            # Handle both tensor and list return types
            f1_vals = bert_results['f1']
            p_vals = bert_results['precision']
            r_vals = bert_results['recall']
            
            # Convert to numpy for robust NaN handling
            if hasattr(f1_vals, 'mean'):  # Is tensor
                f1_np = f1_vals.cpu().numpy() if hasattr(f1_vals, 'cpu') else np.array(f1_vals)
                precision_np = p_vals.cpu().numpy() if hasattr(p_vals, 'cpu') else np.array(p_vals)
                recall_np = r_vals.cpu().numpy() if hasattr(r_vals, 'cpu') else np.array(r_vals)
            else:  # Is list
                f1_np = np.array(f1_vals)
                precision_np = np.array(p_vals)
                recall_np = np.array(r_vals)
            
            # Use nanmean to ignore NaN values (if any remain)
            f1_mean = float(np.nanmean(f1_np))
            p_mean = float(np.nanmean(precision_np))
            r_mean = float(np.nanmean(recall_np))
            
            print(f"  BERTScore F1: {f1_mean:.4f}")
            print(f"  BERTScore Precision: {p_mean:.4f}")
            print(f"  BERTScore Recall: {r_mean:.4f}")
            
            metrics['trans_bertscore_f1'] = f1_mean
            metrics['trans_bertscore_precision'] = p_mean
            metrics['trans_bertscore_recall'] = r_mean

            del bert_scorer, bert_results
            gc.collect()
            
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            import traceback
            traceback.print_exc()
            metrics['trans_bertscore_f1'] = 0.0
            metrics['trans_bertscore_precision'] = 0.0
            metrics['trans_bertscore_recall'] = 0.0
        
        # COMET
        if COMET_AVAILABLE and all_trans_sources:
            try:
                print("Computing COMET scores...")
                # Download and load specific model
                model_path = download_model("Unbabel/wmt22-comet-da")
                comet_model = load_from_checkpoint(model_path)
                
                # Prepare data without filtering (include empty strings for missed bubbles)
                comet_data = [
                    {
                        "src": src,
                        "mt": pred,
                        "ref": ref
                    }
                    for src, pred, ref in zip(all_trans_sources, all_trans_pred, all_trans_expected)
                ]
                
                if comet_data:
                    comet_output = comet_model.predict(
                        comet_data,
                        batch_size=8,
                        gpus=1 if device != 'cpu' else 0
                    )
                    # Use system_score which is the standard aggregate
                    metrics['trans_comet'] = comet_output.system_score
                    print(f"  COMET score: {metrics['trans_comet']:.4f}")
                else:
                    metrics['trans_comet'] = 0.0
                    
                del comet_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Warning: COMET computation failed: {e}")
                import traceback
                traceback.print_exc()
                metrics['trans_comet'] = 0.0
        else:
            if not COMET_AVAILABLE:
                print("Warning: COMET not available, skipping COMET metric")
            elif not all_trans_sources:
                print("Warning: No source texts available for COMET")
            metrics['trans_comet'] = 0.0
            
        # --- Compute Metrics for GT Source Translation (Simulated Perfect OCR) ---
        if all_trans_pred_gt and all_trans_expected_gt:
            print("\nComputing metrics for GT Source Translation...")
            
            # We can reuse the same metric instances
            cer_metric = CharErrorRate()
            wer_metric = WordErrorRate()
            bleu_metric = BLEUScore()
            sacrebleu_metric = SacreBLEUScore()
            chrf_metric = CHRFScore(n_char_order=6, n_word_order=0)
            chrf_pp_metric = CHRFScore(n_char_order=6, n_word_order=2)
            
            bleu_refs_gt = [[ref] for ref in all_trans_expected_gt]
            
            metrics['trans_gt_source_cer'] = cer_metric(all_trans_pred_gt, all_trans_expected_gt).item()
            metrics['trans_gt_source_wer'] = wer_metric(all_trans_pred_gt, all_trans_expected_gt).item()
            metrics['trans_gt_source_bleu'] = bleu_metric(all_trans_pred_gt, bleu_refs_gt).item()
            metrics['trans_gt_source_sacrebleu'] = sacrebleu_metric(all_trans_pred_gt, bleu_refs_gt).item()
            metrics['trans_gt_source_chrf'] = chrf_metric(all_trans_pred_gt, bleu_refs_gt).item()
            metrics['trans_gt_source_chrf_pp'] = chrf_pp_metric(all_trans_pred_gt, bleu_refs_gt).item()
            
            del cer_metric, wer_metric, bleu_metric, sacrebleu_metric, chrf_metric, chrf_pp_metric
            gc.collect()

            # BERTScore for GT Source
            try:
                print("Computing BERTScore (GT Source)...")
                bert_device = 'cpu' if device == 'mps' else device
                bert_scorer = BERTScore(model_name_or_path="microsoft/deberta-xlarge-mnli", device=bert_device)
                bert_scorer.reset()
                bert_scorer.update(all_trans_pred_gt, all_trans_expected_gt)
                bert_results = bert_scorer.compute()
                
                f1_vals = bert_results['f1']
                p_vals = bert_results['precision']
                r_vals = bert_results['recall']
                
                # Convert to numpy
                if hasattr(f1_vals, 'mean'):
                    f1_np = f1_vals.cpu().numpy() if hasattr(f1_vals, 'cpu') else np.array(f1_vals)
                    p_np = p_vals.cpu().numpy() if hasattr(p_vals, 'cpu') else np.array(p_vals)
                    r_np = r_vals.cpu().numpy() if hasattr(r_vals, 'cpu') else np.array(r_vals)
                else:
                    f1_np = np.array(f1_vals)
                    p_np = np.array(p_vals)
                    r_np = np.array(r_vals)

                metrics['trans_gt_source_bertscore_f1'] = float(np.nanmean(f1_np))
                metrics['trans_gt_source_bertscore_precision'] = float(np.nanmean(p_np))
                metrics['trans_gt_source_bertscore_recall'] = float(np.nanmean(r_np))
                
                del bert_scorer, bert_results
                gc.collect()
            except Exception as e:
                print(f"GT Source BERTScore failed: {e}")
                metrics['trans_gt_source_bertscore_f1'] = 0.0
                metrics['trans_gt_source_bertscore_precision'] = 0.0
                metrics['trans_gt_source_bertscore_recall'] = 0.0
                
            # COMET for GT Source - using GT Source text as source
            if COMET_AVAILABLE and all_trans_sources_gt:
                try:
                    print("Computing COMET scores (GT Source)...")
                    # Reuse loaded model if possible, otherwise reload (expensive but safe)
                    model_path = download_model("Unbabel/wmt22-comet-da")
                    comet_model = load_from_checkpoint(model_path)
                    
                    comet_data_gt = [
                        {
                            "src": src,
                            "mt": pred,
                            "ref": ref
                        }
                        for src, pred, ref in zip(all_trans_sources_gt, all_trans_pred_gt, all_trans_expected_gt)
                    ]
                    
                    if comet_data_gt:
                        comet_output_gt = comet_model.predict(
                            comet_data_gt,
                            batch_size=8,
                            gpus=1 if device != 'cpu' else 0
                        )
                        metrics['trans_gt_source_comet'] = comet_output_gt.system_score
                        print(f"  GT Source COMET score: {metrics['trans_gt_source_comet']:.4f}")
                    else:
                        metrics['trans_gt_source_comet'] = 0.0
                        
                    del comet_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"GT Source COMET computation failed: {e}")
                    metrics['trans_gt_source_comet'] = 0.0
            else:
                 metrics['trans_gt_source_comet'] = 0.0
    else:
        metrics['trans_cer'] = 1.0
        metrics['trans_wer'] = 1.0
        metrics['trans_bleu'] = 0.0
        metrics['trans_sacrebleu'] = 0.0
        metrics['trans_chrf'] = 0.0
        metrics['trans_chrf_pp'] = 0.0
        metrics['trans_bertscore_f1'] = 0.0
        metrics['trans_bertscore_precision'] = 0.0
        metrics['trans_bertscore_recall'] = 0.0
        metrics['trans_comet'] = 0.0
        
        # Defaults for GT Source metrics
        metrics['trans_gt_source_cer'] = 1.0
        metrics['trans_gt_source_wer'] = 1.0
        metrics['trans_gt_source_bleu'] = 0.0
        metrics['trans_gt_source_sacrebleu'] = 0.0
        metrics['trans_gt_source_chrf'] = 0.0
        metrics['trans_gt_source_chrf_pp'] = 0.0
        metrics['trans_gt_source_bertscore_f1'] = 0.0
        metrics['trans_gt_source_bertscore_precision'] = 0.0
        metrics['trans_gt_source_bertscore_recall'] = 0.0
        metrics['trans_gt_source_comet'] = 0.0
    
    # Ordering Metrics
    metrics['ordering_kendall_tau'] = np.mean(ordering_taus) if ordering_taus else 0.0
    metrics['ordering_exact_match_rate'] = np.mean(ordering_exact) if ordering_exact else 0.0
    
    # Coverage Metrics
    metrics['gt_coverage_mean'] = np.mean(coverages) if coverages else 0.0
    metrics['bubble_utilization_mean'] = np.mean(utilizations) if utilizations else 0.0
    
    # Detection Statistics (counts and percentages)
    metrics['detection_total_gt_texts'] = total_gt_texts
    metrics['detection_matched_pairs_count'] = total_matched_pairs
    metrics['detection_matched_pairs_pct'] = (total_matched_pairs / total_gt_texts * 100) if total_gt_texts > 0 else 0.0
    metrics['detection_missed_count'] = total_missed_detections
    metrics['detection_missed_pct'] = (total_missed_detections / total_gt_texts * 100) if total_gt_texts > 0 else 0.0
    metrics['detection_false_positive_count'] = total_false_positives
    # Note: False positive percentage is relative to total GT texts for consistency
    metrics['detection_false_positive_pct'] = (total_false_positives / total_gt_texts * 100) if total_gt_texts > 0 else 0.0
    
    # Counts
    metrics['num_pages'] = len(all_results)
    metrics['num_ocr_samples'] = len(all_ocr_pred)
    metrics['num_trans_samples'] = len(all_trans_pred)
    
    return metrics

class OpenMantraDataset(Dataset):
    """Dataset class for OpenMantra manga translation dataset."""
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str = "annotation.json",
        transform: Optional[Any] = None,
        language: str = "ja",
        book_titles: Optional[List[str]] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.language = language
        
        annotation_path = self.root_dir / annotation_file
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        if book_titles is not None:
            self.annotations = [
                book for book in self.annotations 
                if book['book_title'] in book_titles
            ]
        
        self.samples: List[Tuple[Dict, Dict]] = []
        for book in self.annotations:
            book_title = book['book_title']
            for page in book['pages']:
                self.samples.append(({'book_title': book_title}, page))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict, List[Dict]]:
        book_info, page_info = self.samples[idx]
        
        image_paths = page_info.get('image_paths', {})
        relative_image_path = image_paths.get(self.language, '')
        image_path = self.root_dir / relative_image_path
        
        image = Image.open(image_path).convert('RGB')
        image_size = image.size
        
        image_info = {
            'book_title': book_info['book_title'],
            'page_index': page_info.get('page_index', -1),
            'image_path': str(image_path),
            'image_size': image_size,
            'frames': page_info.get('frame', [])
        }
        
        raw_bubbles = page_info.get('text', [])
        bubbles = []
        for bubble in raw_bubbles:
            converted_bubble = {
                'xmin': bubble['x'],
                'ymin': bubble['y'],
                'xmax': bubble['x'] + bubble['w'],
                'ymax': bubble['y'] + bubble['h'],
                'text_ja': bubble.get('text_ja', ''),
                'text_en': bubble.get('text_en', ''),
                'text_zh': bubble.get('text_zh', '')
            }
            bubbles.append(converted_bubble)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, image_info, bubbles
    
    def get_book_titles(self) -> List[str]:
        return list(set(book['book_title'] for book in self.annotations))
