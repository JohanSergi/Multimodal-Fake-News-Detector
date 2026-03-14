import os
import logging
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from sentence_transformers import SentenceTransformer

from dataloader import MultimodalDataset, Modality
from model import JointTextImageModel, JointTextImageDialogueModel, MultimodalFakeNewsDetectionModel

DATA_PATH = "./data"
IMAGES_DIR = os.path.join(DATA_PATH, "images")

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    modality = config.get("modality", "text-image")
    batch_size = config.get("batch_size", 4)
    num_classes = config.get("num_classes", 2)

    test_data_path = config.get("test_data_path")
    preprocessed_test_dataframe_path = config.get("preprocessed_test_dataframe_path")

    trained_model_path = "saved_model/fake_news_model.pth"

    # Load embedder
    text_embedder = SentenceTransformer(config.get("text_embedder"))

    # Image transform
    if Modality(modality) == Modality.TEXT_IMAGE_DIALOGUE:
        image_transform = JointTextImageDialogueModel.build_image_transform()
    else:
        image_transform = JointTextImageModel.build_image_transform()

    # Dataset
    test_dataset = MultimodalDataset(
        from_preprocessed_dataframe=preprocessed_test_dataframe_path,
        data_path=test_data_path,
        modality=modality,
        text_embedder=text_embedder,
        image_transform=image_transform,
        summarization_model=None,
        images_dir=IMAGES_DIR,
        num_classes=num_classes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    logging.info(f"Test dataset size: {len(test_dataset)}")

    # Load model
    hparams = {
        "embedding_dim": 768,
        "num_classes": num_classes
    }

    model = MultimodalFakeNewsDetectionModel(hparams)

    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in test_loader:

            text = batch["text"].to(device)
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            preds, loss = model(text, image, label)

            pred_labels = torch.argmax(preds, dim=1)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\nEvaluation Results")
    print("------------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # Save results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    df = pd.DataFrame([results])
    df.to_csv("evaluation_results.csv", index=False)

    print("\nResults saved to evaluation_results.csv")
