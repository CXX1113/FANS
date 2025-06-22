"""
https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_ImageNet_Example_All_Metrics.ipynb
"""
import quantus
import numpy as np
from config import device

metric_config = {'fashionmnist': {'MaxSens': {'lbd': 0.1, 'abs': True},
                                  'AOPC': {'perturb_baseline': 'black'},
                                  'CORR': {'perturb_baseline': 'black'},
                                  'ROAD': {'perturb_baseline': 'black'}}}


class ExplanationEvaluator:
    def __init__(self, model, dataset_name):
        metric_config2 = metric_config[dataset_name]

        self.infidelity = quantus.Infidelity(
            perturb_baseline="uniform",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            n_perturb_samples=5,
            perturb_patch_sizes=[4],
            return_aggregate=True,
            aggregate_func=np.mean,
            display_progressbar=True,
            disable_warnings=True,)

        self.irof = quantus.IROF(
            segmentation_method="slic",
            perturb_baseline="mean",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            aggregate_func=np.mean,
            return_aggregate=True,
            display_progressbar=True,
            disable_warnings=True,
            )

        self.sparseness = quantus.Sparseness(return_aggregate=True, display_progressbar=True)

        self.model = model
        self.dataset_name = dataset_name

    def report_metrics(self, heatmaps, x_batch, y_batch, method_name, explain_func=None, explain_func_kwargs=None):
        # Return Max-Sensitivity scores for Saliency.
        if explain_func is None:
            explain_func = quantus.explain
        if explain_func_kwargs is None:
            explain_func_kwargs = {'method': method_name}

        infidelity = self.infidelity(model=self.model, x_batch=x_batch, y_batch=y_batch, a_batch=heatmaps,
                                     device=device)[0]

        irof = self.irof(model=self.model, x_batch=x_batch, y_batch=y_batch, a_batch=heatmaps, device=device)

        sparseness = self.sparseness(model=self.model, x_batch=x_batch, y_batch=y_batch,
                                     a_batch=heatmaps, device=device)

        irof = irof[0]
        sparseness = sparseness[0]

        print("=" * 5, f"Explainer={method_name} Dataset={self.dataset_name} Performance Report", "=" * 5)

        # print("|***|Faithfulness|***| Metrics:")
        print(f"Infidelity(-): {infidelity:.4f} ()")
        print(f"IROF(+): {irof:.4f} ()")
        print(f"Sparseness(+): {sparseness:.4f}")
        print(f"{infidelity:.4f}\t{irof:.4f}\t{sparseness:.4f}")
