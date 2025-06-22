from models import lenet_fashionmnist


def load_model(dataset_name, train_loader, test_loader):
    if dataset_name == 'fashionmnist':
        model = lenet_fashionmnist.load_model(train_loader, test_loader)
    else:
        assert False, f"Unknown dataset {dataset_name}"

    return model
