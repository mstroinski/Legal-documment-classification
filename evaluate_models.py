from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import namedtuple

def calculate_metrics(y, y_pred, model_name='', get_metrics=False):
    f1 = f1_score(y, y_pred, average='weighted')
    precission = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    accuracy = accuracy_score(y, y_pred)

    print(f"Metryki dla modelu {model_name}:\n\
{accuracy = :.2f}\n\
{f1 = :.2f}\n\
{precission = :.2f}\n\
{recall = :.2f}")

    if get_metrics:
        Metrics = namedtuple('Metrics', ['accuracy', 'f1_score', 'precission', 'recall'])
        metrics = Metrics(accuracy=accuracy, f1_score=f1, precission=precission, recall=recall)
        
        return metrics


