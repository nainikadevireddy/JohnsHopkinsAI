import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

class Metrics:
    """
    Metrics class for evaluating binary classification models.
    """

    def __init__(self, model, X_eval, y_eval):
        """
        Initialize the Metrics class with a model and evaluation dataset.

        Parameters:
        model (BaseEstimator): The trained model to evaluate.
        X_eval (DataFrame or ndarray): The evaluation features.
        y_eval (Series or ndarray): The evaluation labels.
        """
        self.model = model
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.predicted_labels = None
        self.predicted_probabilities = None
        self.evaluate()

    def evaluate(self):
        """
        Evaluate the model using the specified evaluation data.
        """
        self.predicted_labels = self.model.predict(self.X_eval)
        if hasattr(self.model, "predict_proba"):
            self.predicted_probabilities = self.model.predict_proba(self.X_eval)[:, 1]
        elif hasattr(self.model, "decision_function"):
            self.predicted_probabilities = self.model.decision_function(self.X_eval)
        else:
            self.predicted_probabilities = None


    def accuracy(self):
        """
        Calculate the accuracy of the model.

        Returns:
        float: The accuracy score.
        """
        return accuracy_score(self.y_eval, self.predicted_labels)

    def precision(self):
        """
        Calculate the precision of the model.

        Returns:
        float: The precision score.
        """
        return precision_score(self.y_eval, self.predicted_labels)

    def recall(self):
        """
        Calculate the recall of the model.

        Returns:
        float: The recall score.
        """
        return recall_score(self.y_eval, self.predicted_labels)

    def f1(self):
        """
        Calculate the F1 score of the model.

        Returns:
        float: The F1 score.
        """
        return f1_score(self.y_eval, self.predicted_labels)

    def roc_auc(self):
        """
        Calculate the ROC-AUC score of the model.

        Returns:
        float: The ROC-AUC score.
        """
        if self.predicted_probabilities == None: 
            return None
        else: 
            return roc_auc_score(self.y_eval, self.predicted_probabilities)

    def run_metrics(self):
        """
        Calculate all the metrics: accuracy, precision, recall, and F1 score.

        Returns:
        dict: A dictionary containing all the metrics.
        """
        return {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1(),
            'roc_auc': self.roc_auc()
        }
