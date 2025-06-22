import numpy as np
import quantus
import torch
import torchvision
from config import root_dir
from model import load_model
from evaluate import ExplanationEvaluator
from torch.utils.data import DataLoader
import explainer

seed = 1  # 1, 45

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = 'FashionMNIST'  # MNIST CIFAR10 FashionMNIST

method_name = 'FANS'
test_size = 200  # 200, 4000

# Load datasets and make loaders.

dataset_name = dataset_name.lower()
# method_name = method_name.lower()

# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

classes_map = None
if dataset_name == 'fashionmnist':
    transformer = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.FashionMNIST(root=root_dir, train=True, transform=transformer, download=True)
    test_set = torchvision.datasets.FashionMNIST(root=root_dir, train=False, transform=transformer, download=True)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=torch.cuda.is_available())  # num_workers=4,
    test_loader = DataLoader(test_set, batch_size=test_size, pin_memory=torch.cuda.is_available())

    # Specify class labels.
    classes_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
                   5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

    # Load a batch of inputs and outputs to use for evaluation.
    x_batch, y_batch = next(iter(test_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
else:
    assert False, f"Unknown dataset {dataset_name}"


model = load_model(dataset_name, train_loader, test_loader).to(device)

# Generate explanations
explain_func, explain_func_kwargs = None, None

# init_method = 'IntegratedGradients' if dataset_name == 'mnist' else 'Occlusion'
init_method = 'IntegratedGradients'
attribute_batch = quantus.explain(model, x_batch, y_batch, method=init_method)
attribute_batch = attribute_batch / x_batch.shape[1]
# attribute_batch = None
explain_func_kwargs = {'dataset_name': dataset_name, 'init_masks': attribute_batch}
attribute_batch = explainer.explain(model, x_batch, y_batch, **explain_func_kwargs)
explain_func = explainer.explain


# Save x_batch and y_batch as numpy arrays that will be used to call metric instances.
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

# Quick assert.
assert [isinstance(obj, np.ndarray) for obj in [x_batch, y_batch, attribute_batch]]

evaluator = ExplanationEvaluator(model, dataset_name)

evaluator.report_metrics(attribute_batch, x_batch, y_batch, method_name, explain_func, explain_func_kwargs)
