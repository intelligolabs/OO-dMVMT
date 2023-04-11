import numpy as np
import matplotlib.pyplot as plt
import torch
import re

# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
#     return sorted(data, key=alphanum_key)

def print_confusion_matrix(y_test, predictions, path='viz/ConfMatrix.jpg'):

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_test, predictions, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(15,15))
    disp.plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.show()
    fig.savefig(path)
    return cm

def test_results(model,loader,device='cuda'):

    preds = []
    gt = []

    model .eval()
    with torch.no_grad():
        n = 0
        for cnt,(batch,label) in enumerate(loader):

            n+=1

            batch=batch.to(device)
            label=label.to(device)

            output=model(batch)
            preds.append(output.cpu().detach().numpy().argmax(axis=1).tolist())
            gt.append(label.cpu().detach().numpy().tolist())

    preds = [p for sublist in preds for p in sublist]
    gt = [p for sublist in gt for p in sublist]

    print_confusion_matrix(preds,gt)

