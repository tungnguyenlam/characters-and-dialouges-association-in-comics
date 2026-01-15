# code/pipeline/Evaluators/evals_utils/TranslationEvaluator.py

from abc import abstractmethod
from typing import Dict, List, Optional, Any
import gc
from tqdm.auto import tqdm
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore, SacreBLEUScore, CHRFScore
from torchmetrics.text.bert import BERTScore
from Evaluator import Evaluator
import torch
try:
    from evals_utils.utils import save_output_to_json, save_metrics_to_json, print_tabulate_results, get_run_dir
except ImportError:
    from ..evals_utils.utils import save_output_to_json, save_metrics_to_json, print_tabulate_results, get_run_dir

# Optional COMET import
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
class TranslationEvaluator(Evaluator):
    """
    Abstract base class for all translation evaluators.
    
    Provides common functionality for computing translation metrics:
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - BLEU Score
    - SacreBLEU Score
    - chrF Score
    - chrF++ Score
    
    Subclasses must implement:
    - get_dataset(): Load and return the specific evaluation dataset
    - get_dataloader(): Create and return a DataLoader for the dataset
    """
    
    def __init__(self, hf_token: Optional[str] = None, session_name: Optional[str] = None):
        super().__init__()
        self.hf_token = hf_token
        self.session_name = session_name
    
    @abstractmethod
    def get_dataset(self) -> Any:
        """
        Load and return the evaluation dataset.
        
        Returns:
            Dataset object compatible with PyTorch DataLoader.
        """
        pass
    
    @abstractmethod
    def get_dataloader(self, batch_size: int = 1) -> Any:
        """
        Create and return a DataLoader for the evaluation dataset.
        
        Args:
            batch_size: Batch size for evaluation.
            
        Returns:
            PyTorch DataLoader.
        """
        pass

    def save_output_to_json(
        self, 
        source_texts: List[str], 
        expected: List[str], 
        predicted: List[str], 
        file_dir: str,
        model_name: Optional[str] = None,
        create_run_folder: bool = True
    ):
        save_output_to_json(source_texts, expected, predicted, file_dir, model_name, create_run_folder)

    def save_metrics_to_json(
        self, 
        metrics: Dict[str, float], 
        file_dir: str,
        model_name: Optional[str] = None,
        create_run_folder: bool = False
    ):
        save_metrics_to_json(metrics, file_dir, model_name, create_run_folder)

    def print_tabulate_results(self, metrics: Dict[str, float]):
        print_tabulate_results(metrics)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != 'auto':
            return device
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def compute_metrics(
        self, 
        predicted: List[str], 
        expected: List[str],
        source_texts: Optional[List[str]] = None,
        device: str = 'auto',
        use_neural_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Compute all translation metrics.
        
        Args:
            predicted: List of predicted translations.
            expected: List of ground truth translations.
            source_texts: List of source texts (required for COMET).
            device: Device for BERTScore computation.
            use_neural_metrics: If True, compute BERTScore and COMET (default: True).
            
        Returns:
            Dictionary with all metric values.
        """
        device = self._resolve_device(device)

        # CER
        metric_cer = CharErrorRate()
        cer = metric_cer(predicted, expected)
        
        # WER
        metric_wer = WordErrorRate()
        wer = metric_wer(predicted, expected)
        
        # Format references for BLEU-style metrics: [[ref1], [ref2], ...]
        bleu_formatted_refs = [[ref] for ref in expected]
        
        # BLEU
        metric_bleu = BLEUScore()
        bleu = metric_bleu(predicted, bleu_formatted_refs)
        
        # SacreBLEU
        sacre_bleu = SacreBLEUScore()
        sacre_bleu_score = sacre_bleu(predicted, bleu_formatted_refs)
        
        # chrF
        metric_chrf = CHRFScore(n_char_order=6, n_word_order=0)
        chrf = metric_chrf(predicted, bleu_formatted_refs)
        
        # chrF++
        metric_chrf_pp = CHRFScore(n_char_order=6, n_word_order=2)
        chrf_pp = metric_chrf_pp(predicted, bleu_formatted_refs)
        
        metrics = {
            "cer": cer.item(),
            "wer": wer.item(),
            "bleu": bleu.item(),
            "sacrebleu": sacre_bleu_score.item(),
            "chrf": chrf.item(),
            "chrf_pp": chrf_pp.item()
        }
        
        # Cleanup string metrics
        del metric_cer, metric_wer, metric_bleu, sacre_bleu, metric_chrf, metric_chrf_pp
        gc.collect()
        
        # Neural metrics (BERTScore and COMET)
        if use_neural_metrics:
            # BERTScore
            try:
                import numpy as np
                print("Computing BERTScore...")
                bert_scorer = BERTScore(
                    model_name_or_path="microsoft/deberta-xlarge-mnli",
                    device=device
                )
                bert_scorer.reset()
                bert_scorer.update(predicted, expected)
                bert_results = bert_scorer.compute()
                
                # Handle both tensor and list return types (varies by torchmetrics version)
                f1_vals = bert_results['f1']
                precision_vals = bert_results['precision']
                recall_vals = bert_results['recall']
                
                # Check if values are tensors (have .mean()) or lists
                if hasattr(f1_vals, 'mean'):
                    # Convert to numpy for NaN filtering
                    f1_np = f1_vals.cpu().numpy() if hasattr(f1_vals, 'cpu') else np.array(f1_vals)
                    precision_np = precision_vals.cpu().numpy() if hasattr(precision_vals, 'cpu') else np.array(precision_vals)
                    recall_np = recall_vals.cpu().numpy() if hasattr(recall_vals, 'cpu') else np.array(recall_vals)
                else:
                    f1_np = np.array(f1_vals)
                    precision_np = np.array(precision_vals)
                    recall_np = np.array(recall_vals)
                
                # Use nanmean to ignore NaN values
                metrics["bertscore_f1"] = float(np.nanmean(f1_np))
                metrics["bertscore_precision"] = float(np.nanmean(precision_np))
                metrics["bertscore_recall"] = float(np.nanmean(recall_np))
                
                del bert_scorer, bert_results
                gc.collect()
            except Exception as e:
                print(f"BERTScore computation failed: {e}")
                import traceback
                traceback.print_exc()
                metrics["bertscore_f1"] = None
                metrics["bertscore_precision"] = None
                metrics["bertscore_recall"] = None
            
            # COMET (requires source texts)
            if COMET_AVAILABLE and source_texts is not None:
                try:
                    print("Computing COMET score...")
                    comet_model_path = download_model("Unbabel/wmt22-comet-da")
                    comet_model = load_from_checkpoint(comet_model_path)
                    
                    # Prepare data for COMET
                    comet_data = [
                        {"src": src, "mt": mt, "ref": ref}
                        for src, mt, ref in zip(source_texts, predicted, expected)
                    ]
                    comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1 if device != 'cpu' else 0)
                    metrics["comet"] = comet_output.system_score
                    
                    del comet_model
                    gc.collect()
                except Exception as e:
                    print(f"COMET computation failed: {e}")
                    metrics["comet"] = None
            elif use_neural_metrics and not COMET_AVAILABLE:
                print("COMET not available. Install with: pip install unbabel-comet")
                metrics["comet"] = None
            elif use_neural_metrics and source_texts is None:
                print("COMET requires source_texts. Skipping COMET.")
                metrics["comet"] = None
        
        return metrics

    def run_inference(self, model, dataloader, run_dir: Optional[str] = None, save_steps: int = 50):
        source_texts = []
        expected = []
        predicted = []
        
        for i, batch in enumerate(tqdm(dataloader)):
            batch_src_texts = batch["src_text"]
            batch_tgt_texts = batch["tgt_text"]
            batch_predicted_texts = model.predict(batch_src_texts)
            source_texts.extend(batch_src_texts)
            expected.extend(batch_tgt_texts)
            predicted.extend(batch_predicted_texts)

            # Save intermediate results (without creating new folder each time)
            if run_dir is not None and i % save_steps == 0:
                self.save_output_to_json(
                    source_texts, expected, predicted, 
                    run_dir, 
                    create_run_folder=False
                )
            
        return source_texts, expected, predicted
    
    def evaluate(
        self,
        model: Any,
        batch_size: int = 1,
        device: str = 'auto',
        verbose: bool = True,
        save_steps: int = 50,
        save_dir: Optional[str] = None,
        use_neural_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a translation model.
        
        Args:
            model: Translation model with predict(List[str]) -> List[str] method.
                   Must also have load_model() and unload_model() methods.
            batch_size: Batch size for evaluation.
            device: Device for computation.
            verbose: If True, print progress and sample predictions.
            save_steps: Save intermediate results every N steps.
            save_dir: Base directory to save outputs. A timestamped subfolder will be created.
                      If None, results are not saved.
            use_neural_metrics: If True, compute BERTScore and COMET (default: True).
            
        Returns:
            Dictionary with metric values.
        """
        from tqdm.auto import tqdm
        
        dataloader = self.get_dataloader(batch_size)

        model.load_model()
        
        model_name = getattr(model, 'model_name', 'unknown')

        # Create run directory at the start (so intermediate saves go to same folder)
        run_dir = None
        if save_dir is not None:
            run_dir = get_run_dir(save_dir, model_name)
            print(f"Saving results to: {run_dir}")
        
        source_texts, expected, predicted = self.run_inference(model, dataloader, run_dir, save_steps)
        
        model.unload_model()
        
        # Compute metrics
        metrics = self.compute_metrics(
            predicted, 
            expected, 
            source_texts=source_texts,
            device=device,
            use_neural_metrics=use_neural_metrics
        )
        metrics["model_name"] = model_name
        metrics["translator_class"] = model.__class__.__name__
        if self.session_name:
            metrics["session_name"] = self.session_name
        
        # Print and save results
        if verbose:
            self.print_tabulate_results(metrics)
        
        if run_dir is not None:
            self.save_output_to_json(
                source_texts, expected, predicted, 
                run_dir, 
                create_run_folder=False
            )
            self.save_metrics_to_json(metrics, run_dir, create_run_folder=False)
        
        return metrics