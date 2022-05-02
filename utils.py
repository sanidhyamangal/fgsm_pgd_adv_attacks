"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import matplotlib.pyplot as plt  # fpr plotting
import torch  # for torch based ops
from torchvision import transforms

DEVICE = lambda: "cuda" if torch.cuda.is_available() else "cpu"


def save_image(image, title, filename: str):
    """function to save images"""
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


def generate_pgd_adv(model,
                     images,
                     y,
                     criterion,
                     eps,
                     alpha,
                     num_iter,
                     targeted: bool = False):
    """function to generate perturbations for adv attack using pdg method"""
    adv_image = images.clone().detach().requires_grad_(
        True)  # copy the adv image from the og image
    for i in range(num_iter):  # iterate for specified num of iterations.
        _adv = adv_image.clone().detach().requires_grad_(
            True)  # create a temp copy of the adv image to compute grads

        # compute gradient and loss
        output = model(_adv)
        model.zero_grad()
        loss = criterion(output, y)
        loss.backward()

        # compute the sign of the grad
        grad = torch.sign(_adv.grad)

        # check if the attack is targetd or not
        if not targeted:
            adv_image = adv_image + grad * alpha
        else:
            adv_image = adv_image - grad * alpha

        # perform projected gradient op for the training part
        adv_image = torch.max(torch.min(adv_image, images + eps), images - eps)
        adv_image = adv_image.clamp(0.0, 1.0)

    # return ad
    return adv_image.detach(), adv_image - images


def normalize_image(image: torch.tensor, mean, std):
    """function to normalize image before saving it"""
    return image.detach().mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        torch.FloatTensor(mean).view(3, 1, 1)).detach()


def create_image_transforms(mean, std):
    """function to create transformation pipeline for the machine learning"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_categories(filename):
    """function to get resnet labels"""
    with open(filename, "r") as fp:
        categories = [s.strip() for s in fp.readlines()]

    return categories


def generate_fgsm_pertub(model, input_image, target_label, criterion):
    """function to generate perturbations using FGSM method"""
    # make zero grad
    model.zero_grad()
    # mark require grad to true for image
    input_image.requires_grad = True
    # generate predictions, compute loss and gradients
    predictions = model(input_image)

    loss = criterion(predictions, target_label)
    loss.backward()
    # obtain the grad and return it's sign
    grad = input_image.grad.data

    return torch.sign(grad)
