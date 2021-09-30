import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.onnx
import torch.nn.functional as F
import subprocess


class CNN_MNIST(nn.Module):
    def __init__(self, fc_1_output, dropout_prob):
        super(CNN_MNIST, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(8 * 7 * 7, fc_1_output)
        self.fc2 = nn.Linear(fc_1_output, 10)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flattening the input tensor
        x = x.view(-1, 8 * 7 * 7)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x


def load_model():
    train_on_gpu = torch.cuda.is_available()
    model = CNN_MNIST(fc_1_output=128, dropout_prob=0.2)
    if train_on_gpu:
        model.cuda()
    model.load_state_dict(torch.load("cnn_model_SGD_fasion_mnist.pt"))
    return model


def pytorch_2_onnx():
    BATCH_SIZE = 16

    # a transform for: Tensor and Normalizing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # training data
    train_data = datasets.FashionMNIST(
        "~/.pytorch/F_MNIST_data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        images, labels = images.cuda(), labels.cuda()
    dummy_input = images

    model = load_model()
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        "fmnist_cnn.onnx",
        verbose=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )


def onnx_to_trt():
    subprocess.run(
        "trtexec --onnx=fmnist_cnn.onnx --saveEngine=fmnist_cnn.trt --explicitBatch",
        shell=True,
    )


def test_onnx():
    import onnx
    import onnxruntime
    import torch
    import numpy as np

    onnx_model = onnx.load("fmnist_cnn.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("fmnist_cnn.onnx")

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    train_on_gpu = torch.cuda.is_available()
    model = load_model()
    BATCH_SIZE = 16

    # a transform for: Tensor and Normalizing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    test_data = datasets.FashionMNIST(
        "~/.pytorch/F_MNIST_data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True
    )

    label_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:

            # =============================================================================
            #            if train_on_gpu:
            #                images, labels = images.cuda(), labels.cuda()
            # =============================================================================
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
            ort_outs = ort_session.run(None, ort_inputs)
            ort_outs = torch.Tensor(np.array(ort_outs))
            ort_outs = ort_outs.squeeze(0)
            # print('ONNX MODEL: output shape:', ort_outs.shape)
            test_loss += criterion(torch.Tensor(np.array(ort_outs)), labels)

            ps = torch.exp(ort_outs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(
        "Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
        "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)),
    )


def test_model():
    train_on_gpu = torch.cuda.is_available()
    model = load_model()
    BATCH_SIZE = 16

    # a transform for: Tensor and Normalizing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    test_data = datasets.FashionMNIST(
        "~/.pytorch/F_MNIST_data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True
    )

    label_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # =============================================================================
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            # =============================================================================
            output = model(images)

            test_loss += criterion(output, labels)

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(
        "Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
        "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)),
    )


if __name__ == "__main__":
    # test_model()
    pytorch_2_onnx()
    # test_onnx()
    onnx_to_trt()
