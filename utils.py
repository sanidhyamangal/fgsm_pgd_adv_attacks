"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import matplotlib.pyplot as plt  # fpr plotting
import torch  # for torch based ops
from torchvision import transforms

DEVICE = lambda: "cuda" if torch.cuda.is_available() else "cpu"


def save_image(image, title, filename: str):
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


def random_init(x, eps):
    x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) -
             0.5) * 2 * eps
    x = torch.clamp(x, 0, 1)
    return x


def generate_pgd_adv(model,
                     images,
                     y,
                     criterion,
                     eps,
                     alpha,
                     num_iter,
                     targeted: bool = False):
    adv = images.clone().detach().requires_grad_(True)
    for i in range(num_iter):
        _adv = adv.clone().detach().requires_grad_(True)
        output = model(_adv)
        model.zero_grad()
        loss = criterion(output, y)
        loss.backward()
        grad = _adv.grad
        grad = grad.sign()
        if not targeted:
            adv = adv + grad * alpha
        else:
            adv = adv - grad * alpha

        adv = torch.max(torch.min(adv, images + eps), images - eps)
        adv = adv.clamp(0.0, 1.0)

    return adv.detach(), adv - images


def normalize_image(image: torch.tensor, mean, std):
    return image.detach().mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        torch.FloatTensor(mean).view(3, 1, 1)).detach()


def create_image_transforms(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_categories(filename):
    with open(filename, "r") as fp:
        categories = [s.strip() for s in fp.readlines()]

    return categories


def generate_fgsm_pertub(model, input_image, target_label, criterion):
    model.zero_grad()
    input_image.requires_grad = True
    predictions = model(input_image)

    loss = criterion(predictions, target_label)
    loss.backward()
    grad = input_image.grad.data

    return torch.sign(grad)
