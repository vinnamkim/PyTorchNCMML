import torch
from sklearn import datasets
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pytorch_ncmml.utils import get_class_mean
from pytorch_ncmml.ncmml import LinearTransform, NCMMLLoss
from torch.optim import SGD
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = torch.tensor(iris['data'], dtype=torch.float), torch.tensor(iris['target'], dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10)
mean = get_class_mean(train_dataloader)
print(mean)

transform = LinearTransform(in_features=4, out_features=2, init_method='identity')
criterion = NCMMLLoss()
optimizer = SGD(transform.parameters(), lr=0.1)

for epoch in range(200):
    for batch in train_dataloader:
        features, targets = batch

        features = transform(features)
        features_mean = transform(mean)

        optimizer.zero_grad()
        loss = criterion(features, features_mean, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch} Loss: {loss.detach().item()}')


import matplotlib.pyplot as plt

features, targets = zip(*[batch for batch in test_dataloader])
features = torch.cat(features)
targets = torch.cat(targets)

labeled_features = [features[targets == t] for t in targets.unique()]
colors = ['r', 'g', 'b']
#plt.scatter()

transform.eval()
for idx, l_features in enumerate(labeled_features):
    X = transform(l_features).detach().numpy()
    plt.scatter(X[:, 0], X[:, 1], color=colors[idx])
plt.show()

print('Done')
