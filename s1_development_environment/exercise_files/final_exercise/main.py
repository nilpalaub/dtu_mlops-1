import click
from matplotlib import pyplot as plt
import torch
import sys
from pathlib import Path

# Add the parent directory to sys.path to be able to import helper
sys.path.append(str(Path(__file__).resolve().parent.parent))

import helper  # Now you can import helper

from model import MyAwesomeModel

from data import mnist
from torch.utils.data import DataLoader




@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=20, help="number of epochs to train for")
@click.option("--batch_size", default=256, help="batch size to use for training")
def train(lr, epochs, batch_size):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = mnist()

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in train_data_loader:
            images, labels = batch
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch} Loss: {loss}")

    torch.save(model, "model.pt")
            

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)


    model = torch.load(model_checkpoint)
    _, test_set = mnist()

    model.eval()
    test_data_loader = DataLoader(test_set, batch_size=256, shuffle=True)
    accuracies = []

    with torch.no_grad():
        for batch in test_data_loader:
            images, labels = batch
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            accuracies.append(torch.sum(predictions == labels) / len(labels))

    print(f"Accuracy: {sum(accuracies) / len(accuracies)}")



@click.command()
@click.option("--index", default=0, help="index of the image to show")
def showImage(index):
    """Show a image from the dataset."""
    print("Showing image")

    train_dataset, test_dataset = mnist()
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    image_tensors, labels = next(iter(train_data_loader))

    # plot samething to check if plt is working
    helper.imshow(image_tensors[index], normalize=False)
    

cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(showImage)


if __name__ == "__main__":
    cli()
