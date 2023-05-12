import torch
from torch import nn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, ViTImageProcessor, AutoFeatureExtractor
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = {
    "VIT": {
        "model": AutoModelForImageClassification.from_pretrained(
            "Ahmed9275/Vit-Cifar100"
        ),
        "extractor": ViTImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100"),
    },
    "SWIN": {
        "model": AutoModelForImageClassification.from_pretrained("MazenAmria/swin-small-finetuned-cifar100"),
        "extractor": ViTImageProcessor.from_pretrained("MazenAmria/swin-small-finetuned-cifar100")
    }
}


def transform_function(image):
    image = extractor(images=image, return_tensors="pt")
    image["pixel_values"] = image["pixel_values"].squeeze()
    return image


def get_dataset():
    test_dataset = CIFAR100(
        root="./data", train=False, download=True, transform=transform_function
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )

    return test_dataloader


# define a helper function to calculate the accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="VIT")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model = models[args.model]["model"]
    extractor = models[args.model]["extractor"]

    model.to(device)
    model.eval()

    test_dataloader = get_dataset()

    eval_accuracy = 0
    nb_eval_steps = 0
    for batch in tqdm(test_dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        with torch.no_grad():
            outputs = model(**b_input_ids)

        logits = outputs.logits

        tmp_eval_accuracy = torch.sum(torch.argmax(logits, dim=1) == b_labels) / len(
            b_labels
        )

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Accuracy: {}".format(eval_accuracy / nb_eval_steps))
