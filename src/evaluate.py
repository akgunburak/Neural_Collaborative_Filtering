import numpy as np
import torch


def hit_ratio(test_item, pred_items):
    """

    Parameters
    ----------
    test_item : int
        Item id of the item that will be used for testing.
    pred_items : list
        Items that are ranked by the model.

    Returns
    -------
    int
        Hit ratio.

    """
    if test_item in pred_items:
        return 1
    return 0


def ndcg(test_item, pred_items):
    """

    Parameters
    ----------
    test_item : int
        Item id of the item that will be used for testing.
    pred_items : list
        Items that are ranked by the model.

    Returns
    -------
    int
        Normalized discounted cumulative gain ratio.
    """
    if test_item in pred_items:
        index = pred_items.index(test_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(model, test_loader, top_k, device):
    """

    Parameters
    ----------
    model : Torch model
        Model to be evaluated.
    test_loader : torch.utils.data.DataLoader
        Torch DataLoader object.
    top_k : int
        Top k instances to calculate metrics.
    device : torch.device
        Torch device object.

    Returns
    -------
    Float
        Average of HR and NDCG values.

    """
    hr_list, ndcg_list = [], []

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        test_item = item[0].item()  # Leave one-out evaluation has only one item per user
        
        hr_list.append(hit_ratio(test_item, recommends))
        ndcg_list.append(ndcg(test_item, recommends))
        
    return np.mean(hr_list), np.mean(ndcg_list)
