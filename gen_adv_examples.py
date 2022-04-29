"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import torch
import torch.nn as nn
from PIL import Image

from utils import (create_image_transforms, generate_pgd_adv, get_categories,
                   normalize_image, save_image)

image = Image.open("YellowLabradorLooking_new.jpeg")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()
categories = get_categories("imagenet_classes.txt")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = create_image_transforms(mean, std)

image_tensor = preprocess(image)
criterion = nn.CrossEntropyLoss()

# generate first set of predictions
logits = model(image_tensor.unsqueeze(0))
probability, category_id = torch.topk(torch.sigmoid(logits), 1)

print(f"Probabilities: {probability}, category: {category_id}")

# generate untrageted_samples
pgd_un_adv_image, pgd_pertub = generate_pgd_adv(
    model=model,
    images=image_tensor.unsqueeze(0),
    y=category_id[0],
    criterion=criterion,
    eps=1e-2,
    alpha=1e-4,
    num_iter=50)
pgd_targ_adv_image, pgd_pertub_targ = generate_pgd_adv(
    model,
    image_tensor.unsqueeze(0),
    torch.LongTensor([9]),
    criterion=criterion,
    eps=1e-2,
    alpha=1e-4,
    num_iter=50,
    targeted=True)

pgd_un_prob = model(pgd_un_adv_image)
pgd_targ_prob = model(pgd_targ_adv_image)

pgd_un_probab, pgd_un_cat_id = torch.topk(torch.sigmoid(pgd_un_prob), 1)
pgd_targ_probab, pgd_targ_cat_id = torch.topk(torch.sigmoid(pgd_targ_prob), 1)

print(
    f"Probab Un {pgd_un_probab}, Cat Id: {pgd_un_cat_id} Category: {categories[pgd_un_cat_id]}"
)
print(
    f"Probab Targ {pgd_targ_probab}, Cat Id: {pgd_targ_cat_id} Category: {categories[pgd_targ_cat_id]}"
)

save_image(
    normalize_image(pgd_un_adv_image, mean, std).numpy().transpose(1, 2, 0),
    "PGD Untargetd Adv Image", "pgd_un_adv_img.png")
save_image(pgd_pertub.detach().numpy().transpose(1, 2, 0),
           "PGD Untargetd Pertubed Image", "pgd_un_per_img.png")

save_image(
    normalize_image(pgd_targ_adv_image, mean, std).numpy().transpose(1, 2, 0),
    "PGD Targetd Adv Image", "pgd_targ_adv_img.png")
save_image(pgd_pertub_targ.detach().numpy().transpose(1, 2, 0),
           "PGD Targetd Pertubed Image", "pgd_targ_per_img.png")
