import numpy as np


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.1, max_loss_diff=0.2):
        self.patience = patience
        self.min_delta = min_delta
        self.max_diff = max_loss_diff
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, train_loss):
        if self.loss_val_is_lower(validation_loss) and not self.is_overfitting(train_loss, validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif self.loss_val_is_higher(validation_loss) or self.is_overfitting(train_loss, validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def loss_val_is_lower(self, validation_loss):
        return validation_loss < self.min_validation_loss

    def is_overfitting(self, train_loss, validation_loss):
        return validation_loss - train_loss >= self.max_diff

    def loss_val_is_higher(self, validation_loss):
        return validation_loss > (self.min_validation_loss + self.min_delta)


