"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import torch
import torch.nn as nn
from PIL import Image

from utils import (create_image_transforms, generate_fgsm_pertub,
                   generate_pgd_adv, get_categories, normalize_image,
                   save_image)

# load image and model
image = Image.open("YellowLabradorLooking_new.jpeg")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()
# obtain categories for the labels
categories = get_categories("imagenet_classes.txt")
# define the mean and std for normalizing images
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = create_image_transforms(mean, std)

image_tensor = preprocess(image)
criterion = nn.CrossEntropyLoss()  # loss criterion

# generate first set of predictions
logits = model(image_tensor.unsqueeze(0))
probability, category_id = torch.topk(torch.sigmoid(logits), 1)

print(f"Probabilities: {probability}, category: {category_id}")

# generate untrageted_samples using pgd approach
pgd_un_adv_image, pgd_pertub = generate_pgd_adv(
    model=model,
    images=image_tensor.unsqueeze(0),
    y=category_id[0],
    criterion=criterion,
    eps=1e-2,
    alpha=1e-4,
    num_iter=5)

# generate targeted samples for the fgsm approach
pgd_targ_adv_image, pgd_pertub_targ = generate_pgd_adv(
    model,
    image_tensor.unsqueeze(0),
    torch.LongTensor([9]),
    criterion=criterion,
    eps=1e-2,
    alpha=1e-3,
    num_iter=5,
    targeted=True)

# predict the output for the images generated using pgd attack
pgd_un_prob = model(pgd_un_adv_image)
pgd_targ_prob = model(pgd_targ_adv_image)

# generate class probability and category for the predictions
pgd_un_probab, pgd_un_cat_id = torch.topk(torch.sigmoid(pgd_un_prob), 1)
pgd_targ_probab, pgd_targ_cat_id = torch.topk(torch.sigmoid(pgd_targ_prob), 1)

# print results and save images
print(
    f"Probab Un {pgd_un_probab}, Cat Id: {pgd_un_cat_id} Category: {categories[pgd_un_cat_id]}"
)
print(
    f"Probab Targ {pgd_targ_probab}, Cat Id: {pgd_targ_cat_id} Category: {categories[pgd_targ_cat_id]}"
)

save_image(
    normalize_image(pgd_un_adv_image[0], mean, std).numpy().transpose(1, 2, 0),
    "PGD Untargetd Adv Image", "pgd_un_adv_img.png")
save_image(pgd_pertub[0].detach().numpy().transpose(1, 2, 0),
           "PGD Untargetd Pertubed Image", "pgd_un_per_img.png")

save_image(
    normalize_image(pgd_targ_adv_image[0], mean,
                    std).numpy().transpose(1, 2, 0), "PGD Targetd Adv Image",
    "pgd_targ_adv_img.png")
save_image(pgd_pertub_targ[0].detach().numpy().transpose(1, 2, 0),
           "PGD Targetd Pertubed Image", "pgd_targ_per_img.png")

# generate fgsm pertub
fgsm_pertub_un = generate_fgsm_pertub(model, image_tensor.unsqueeze(0),
                                      category_id[0], criterion)
# get adv image
adv_fgsm_un_image = image_tensor + 1e-2 * fgsm_pertub_un
fgsm_un_probab = model(adv_fgsm_un_image)

# get the class probability and category id
fgsm_un_prob, fgsm_un_cat_id = torch.topk(torch.sigmoid(fgsm_un_probab), 1)
# print the results and save images
print(
    f"FGSM: Probab Un {fgsm_un_prob}, Cat Id: {fgsm_un_cat_id} Category: {categories[fgsm_un_cat_id]}"
)
save_image(
    normalize_image(adv_fgsm_un_image[0], mean,
                    std).numpy().transpose(1, 2, 0),
    "FGSM Untargetd Adv Image", "fgsm_un_adv_img.png")
save_image(fgsm_pertub_un[0].detach().numpy().transpose(1, 2, 0),
           "FGSM Untargetd Pertubed Image", "fgsm_un_per_img.png")

# generate fgsm pertub
fgsm_pertub_targ = generate_fgsm_pertub(model, image_tensor.unsqueeze(0),
                                        torch.LongTensor([243]), criterion)
# get adv image
adv_fgsm_targ_image = image_tensor - 1e-2 * fgsm_pertub_targ
fgsm_targ_probab = model(adv_fgsm_targ_image)

# get the class probability and category id
fgsm_targ_prob, fgsm_targ_cat_id = torch.topk(torch.sigmoid(fgsm_targ_probab),
                                              1)
# print the results and save images
print(
    f"FGSM: Probab Targ {fgsm_targ_prob}, Cat Id: {fgsm_targ_cat_id} Category: {categories[fgsm_targ_cat_id]}"
)
save_image(
    normalize_image(adv_fgsm_targ_image[0], mean,
                    std).numpy().transpose(1, 2, 0), "FGSM targetd Adv Image",
    "fgsm_targ_adv_img.png")
save_image(fgsm_pertub_targ[0].detach().numpy().transpose(1, 2, 0),
           "FGSM targetd Pertubed Image", "fgsm_targ_per_img.png")
