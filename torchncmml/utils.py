import torch
from torch.utils.data import DataLoader


def get_class_mean(dataloader: DataLoader) -> torch.Tensor:
    dict_featrues = {}

    for batch in dataloader:
        if isinstance(batch, tuple) or isinstance(batch, list):
            features = batch[0]
            targets = batch[1]
        else:
            raise Exception('batch should be a list of [features, targets].')

        for f, t in zip(features, targets):
            t = t.item()
            if t not in dict_featrues:
                dict_featrues[t] = []

            dict_featrues[t].append(f)

    return torch.stack([torch.stack(dict_featrues[t]).mean(0) for t in sorted(dict_featrues.keys())])


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = torch.tensor(iris['data']), torch.tensor(iris['target'])
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(X, y)

    dataloader = DataLoader(dataset, batch_size=4)
    mean = get_class_mean(dataloader)
    print(mean)
