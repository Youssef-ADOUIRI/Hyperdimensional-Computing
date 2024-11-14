#!/usr/bin/env python
# coding: utf-8
from encoder import RandomFourierEncoder
import torch
import numpy as np
import time
import os
from utils import prepare_data, encode_and_save
from model import BModel, GModel
import argparse


def test(MODEL, loader, criterion, device, model_="rff-hdc"):
    MODEL.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            if model_ == "rff-hdc":
                outputs = MODEL(2 * inputs - 1)
            else:
                outputs = MODEL(inputs)
            test_loss += criterion(outputs, labels)
            preds = outputs.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += preds.eq(labels.view_as(preds)).sum().item()
    test_loss /= len(loader.dataset)
    print(
        "\nTesting Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(loader.dataset),
            100.0 * correct / len(loader.dataset),
        )
    )


def test_with_centroids(
    testloader, centroids, model, device, encoder, metric="manhattan"
):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            encoded_inputs = model(inputs)

            # Classify based on closest centroid
            for i in range(encoded_inputs.size(0)):
                sample = encoded_inputs[i]
                distances = []

                for c, centroid in centroids.items():
                    if metric == "manhattan":
                        distance = encoder.similarity_manhatten(sample, centroid)
                    elif metric == "cosine":
                        distance = encoder.similarity(sample, centroid)
                    distances.append((distance, c))

                # Choose the class with the minimum distance for Manhattan
                if metric == "manhattan":
                    predicted_class = min(distances, key=lambda x: x[0])[1]
                elif metric == "cosine":  # to change it later
                    predicted_class = max(distances, key=lambda x: x[0])[1]

                if predicted_class == labels[i].item():
                    correct += 1
                total += 1

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def compute_centroids(trainloader, model, device, num_classes, encoder):
    centroids = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            encoded_inputs = model(inputs)  # Encode inputs using the model
            for i, label in enumerate(labels):
                centroids[label.item()].append(encoded_inputs[i].cpu())

    # Use group_bundle to calculate the centroid for each class
    for c in centroids:
        if len(centroids[c]) > 0:
            centroids[c] = encoder.group_bundle(torch.stack(centroids[c]).to(device))
        else:
            centroids[c] = torch.zeros(
                encoded_inputs.shape[1], device=device
            )  # Placeholder if no data for class

    return centroids


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    channels = 3 if args.dataset == "cifar" else 1
    classes = (
        26 if args.dataset == "isolet" else (6 if args.dataset == "ucihar" else 10)
    )

    # Initialize the model based on the model type
    model = (
        BModel(in_dim=channels * args.dim, classes=classes).to(device)
        if "hdc" in args.model
        else GModel(args.gorder, in_dim=channels * args.dim, classes=classes).to(device)
    )

    trainloader, testloader = prepare_data(args)

    # Compute class centroids after encoding training data
    print("Calculating centroids for each class...")
    encoder = RandomFourierEncoder(
        input_dim=channels * args.dim,
        gamma=args.gamma,
        gorder=args.gorder,
        output_dim=args.dim,
    )
    encoder.item_mem = torch.load(f"{args.data_dir}/item_mem.pt")
    centroids = compute_centroids(
        trainloader, model, device, num_classes=classes, encoder=encoder
    )

    # Optionally store centroids for use in testing
    torch.save(centroids, f"{args.data_dir}/centroids.pt")

    test_with_centroids(
        testloader, centroids, model, device, metric="manhattan", encoder=encoder
    )


def argument_parser():
    parser = argparse.ArgumentParser(description="HDC Encoding and Training")
    parser.add_argument(
        "-lr",
        type=float,
        default=0.01,
        help="learning rate for optimizing class representative",
    )
    parser.add_argument(
        "-gamma",
        type=float,
        default=0.3,
        help="kernel parameter for computing covariance",
    )
    parser.add_argument("-epoch", type=int, default=1, help="epochs of training")
    parser.add_argument(
        "-gorder",
        type=int,
        default=8,
        help="order of the cyclic group required for G-VSA",
    )
    parser.add_argument(
        "-dim", type=int, default=10000, help="dimension of hypervectors"
    )
    parser.add_argument(
        "-seed", type=int, default=43, help="random seed for reproducing results"
    )
    parser.add_argument(
        "-resume", action="store_true", help="resume from existing encoded hypervectors"
    )
    parser.add_argument(
        "-data_dir",
        default="./encoded_data",
        type=str,
        help="Directory used to save encoded data (hypervectors)",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        choices=["mnist", "fmnist", "cifar", "isolet", "ucihar"],
        default="mnist",
        help="dataset (mnist | fmnist | cifar | isolet | ucihar)",
    )
    parser.add_argument(
        "-raw_data_dir",
        default="./dataset",
        type=str,
        help="Raw data directory to the dataset",
    )
    parser.add_argument(
        "-model",
        type=str,
        choices=["rff-hdc", "linear-hdc", "rff-gvsa"],
        default="rff-gvsa",
        help="feature and model to use: (rff-hdc | linear-hdc | rff-gvsa)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if "hdc" in args.model:
        args.gorder = 2
        print("Use binary HDC with random fourier features, ignoring gorder, set to 2.")
    args.data_dir = f"{args.data_dir}/{args.dataset}_{args.model}_order{args.gorder}_gamma{args.gamma}_dim{args.dim}"
    try:
        os.makedirs(args.data_dir)
    except FileExistsError:
        print("Encoded data folder already exists")
    if not args.resume:
        print("Encode the dataset into hypervectors and save")
        encode_and_save(args)
        print("Finish encoding and saving")
    print(f"Optimizing class representatives for {args.epoch} epochs")
    train(args)
