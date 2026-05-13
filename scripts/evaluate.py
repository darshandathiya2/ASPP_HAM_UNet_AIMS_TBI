
import numpy as np
from sklearn.metrics import precision_score, recall_score

def evaluate_model(model, x_test, y_test):

    predictions = model.predict(x_test)
    predictions = (predictions > 0.5).astype(np.uint8)

    precision = precision_score(
        y_test.flatten(),
        predictions.flatten()
    )

    recall = recall_score(
        y_test.flatten(),
        predictions.flatten()
    )

    return precision, recall
