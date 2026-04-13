import numpy as np
from sklearn.metrics import auc


def auac(accuracies, confidences):    
    confidence_grid = np.arange(0, 1, 0.05)    
    cum_acc = []

    # Calculate cumulative accuracy for each confidence grid point
    for grid in confidence_grid:        
        indices = [i for i, conf in enumerate(confidences) if conf >= grid]

        if indices:            
            correct_predictions = np.sum([accuracies[i].astype(int) for i in indices])
            total_predictions = len(indices)            
            cum_acc.append(correct_predictions / total_predictions)
        else:
            cum_acc.append(0)

    try:
        # out = np.mean(cum_acc)
        out = np.trapz(cum_acc, confidence_grid)
    except:
        # import pdb

        # pdb.set_trace()
        out = None

    return out