import torch as th
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

class LossErrorAccuracyPrecisionRecallF1Metric(Metric):
    """
    Custom metric for calculating loss, error, accuracy, precision, recall, F1 score, and AUC.

    Attributes:
        model (nn.Module): The model being evaluated.
        device (str): The device on which computations are performed.
    """
    def __init__(self, model, just_features, mode, device="cpu", misc_save_path=None):
        self.model = model
        self.just_features = just_features
        self.mode = mode
        self.misc_save_path = misc_save_path
        self._loss_sum = 0
        self._error_sum = 0
        self._accuracy_sum = 0
        self._true_positives = 0
        self._true_negatives = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._num_examples = 0
        self._y_true = [] 
        self._y_scores = []
        self.names_of_false_positives = []
        self.names_of_false_negatives = []
        super().__init__(device=device)

    def reset(self):
        """
        Resets all metric counters to zero.
        """
        self._loss_sum = 0
        self._error_sum = 0
        self._accuracy_sum = 0
        self._true_positives = 0
        self._true_negatives = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._num_examples = 0
        self._y_true = []
        self._y_scores = []
        super().reset()

    def update(self, batch):
        """
        Updates the metric counters based on the current batch.

        Args:
            batch (tuple): A tuple containing the input data and labels.
        """
        if self.just_features == False:
            bag, label = batch[0], batch[1]
            y_bag_true = label[0].float()
        else:
            bag, label, cls, dict = batch
            y_bag_true = label
            batch_name = dict['case_name'][0]
            
        y_bag_pred, _ = self.model(bag)
  
        y_bag_pred = th.clamp(y_bag_pred, min=1e-4, max=1.0 - 1e-4)
        
        loss = th.nn.BCELoss()(y_bag_pred, y_bag_true)
        
        y_bag_pred_binary = th.where(y_bag_pred > 0.5, 1, 0)

        accuracy = th.mean((y_bag_pred_binary == y_bag_true).float()).item()
        error = 1.0 - accuracy

        # Precision, Recall, F1
        self._true_positives += th.sum((y_bag_pred_binary == 1) & (y_bag_true == 1)).item()
        self._true_negatives += th.sum((y_bag_pred_binary == 0) & (y_bag_true == 0)).item()
        self._false_positives += th.sum((y_bag_pred_binary == 1) & (y_bag_true == 0)).item()
        self._false_negatives += th.sum((y_bag_pred_binary == 0) & (y_bag_true == 1)).item()

        # add names of false positives and false negatives
        if self.mode == "test":
            for i in range(len(y_bag_pred_binary)):
                if y_bag_pred_binary[i] == 1 and y_bag_true[i] == 0:
                    self.names_of_false_positives.append(batch_name)
                elif y_bag_pred_binary[i] == 0 and y_bag_true[i] == 1:
                    self.names_of_false_negatives.append(batch_name)

        self._loss_sum += loss.item()
        self._error_sum += error
        self._accuracy_sum += accuracy
        self._num_examples += 1

        self._y_true.extend(y_bag_true.cpu().numpy())
        self._y_scores.extend(y_bag_pred.detach().cpu().numpy())

    def compute(self):
        """
        Computes the final metric values.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        if self._num_examples == 0:
            raise NotComputableError("LossErrorAccuracyPrecisionRecallF1Metric must have at least one example before it can be computed.")
        
        avg_loss = self._loss_sum / self._num_examples
        avg_error = self._error_sum / self._num_examples
        avg_accuracy = self._accuracy_sum / self._num_examples
        
        precision = self._true_positives / (self._true_positives + self._false_positives) if (self._true_positives + self._false_positives) > 0 else 0.0
        recall = self._true_positives / (self._true_positives + self._false_negatives) if (self._true_positives + self._false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        try:
            auc = roc_auc_score(self._y_true, self._y_scores)

        except ValueError:
            auc = float('nan')

        if self.mode == "val":
            return {
                "val/loss": avg_loss,
                "val/error": avg_error,
                "val/accuracy": avg_accuracy,
                "val/precision": precision,
                "val/recall": recall,
                "val/f1": f1_score,
                "val/auc": auc,
                "val/TP": self._true_positives,
                "val/TN": self._true_negatives,
                "val/FP": self._false_positives,
                "val/FN": self._false_negatives
            }
        elif self.mode == "test":
            plot_roc_curve(self._y_true, self._y_scores, self.misc_save_path)
            plot_precision_recall_curve(self._y_true, self._y_scores, self.misc_save_path)
            return {
                "test/loss": avg_loss,
                "test/error": avg_error,
                "test/accuracy": avg_accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1_score,
                "test/auc": auc,
                "test/TP": self._true_positives,
                "test/TN": self._true_negatives,
                "test/FP": self._false_positives,
                "test/FN": self._false_negatives,
                "test/false_positives": self.names_of_false_positives,
                "test/false_negatives": self.names_of_false_negatives,
            }
        

def plot_roc_curve(y_true, y_scores, misc_save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Chance Level (AUC = 0.50)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{misc_save_path}/roc_auc_curve.png")

def plot_precision_recall_curve(y_true, y_scores, misc_save_path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{misc_save_path}/precision_recall_curve.png")