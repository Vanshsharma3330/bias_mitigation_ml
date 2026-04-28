from sklearn.metrics import accuracy_score, f1_score

def evaluate(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)