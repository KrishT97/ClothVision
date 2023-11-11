import pandas as pd
import torch
from matplotlib import pyplot as plt


class EarlyStopper:
    def __init__(self, patience=0, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ModelTrainer:

    def __init__(self, model, optimizer, loss_fn, epochs, early_stopper, device):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result = pd.DataFrame(columns=["Epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        self.early_stopper = early_stopper

    def train_and_val(self, training_loader, val_loader, silent=False):
        for i in range(1, self.epochs + 1):
            train_loss, train_acc = self.train(training_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.result.loc[len(self.result)] = (i,) + (train_loss, train_acc) + (val_loss, val_acc)
            if not silent:
                print("Epoch: " + str(i) + ", Train acc: " + str(train_acc) + ", loss: " + str(
                    train_loss) + " //// Val acc: " + str(val_acc) + " loss: " + str(val_loss))
            if self.early_stopper.early_stop(val_loss):
                self.epochs = i + 1
                break

    def train(self, training_loader):
        running_loss = 0.
        acc_sum = 0
        self.model.train()
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # Make predictions for this batch
            outputs = self.model(inputs).to(self.device)
            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Gather data and report
            running_loss += loss.item()
            acc_sum += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        train_loss = running_loss / len(training_loader)  # perdida media de la época por batch
        acc = 100 * acc_sum / len(training_loader.dataset)  # acc media de la época
        return train_loss, acc

    def validate(self, val_loader):
        self.model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.loss_fn(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * correct / len(val_loader.dataset)
        return val_loss, val_acc

    def draw_results(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.result["Epoch"], self.result["train_loss"], label='Training Loss', marker='o')
        plt.plot(self.result["Epoch"], self.result["val_loss"], label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        # Plotting Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.result["Epoch"], self.result["train_acc"], label='Training Accuracy', marker='o')
        plt.plot(self.result["Epoch"], self.result["val_acc"], label='Validation Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylim(40, 100)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Show the plot
        plt.show()

    def test(self, test_loader):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.loss_fn(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        test_loss /= len(test_loader)
        test_acc = 100 * correct / len(test_loader.dataset)
        print(f"Test Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f} \n")






