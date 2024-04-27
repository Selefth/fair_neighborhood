
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from scipy.sparse import csr_matrix, lil_matrix
from torch.utils.data import Dataset, DataLoader
import random

# Implementation of the MF (Matrix Factorization) model with the non-parity regularizer
# From the paper "Beyond Parity: Fairness Objectives for Collaborative Filtering" presented at NeurIPS 2017

class InteractionDataset(Dataset):
    def __init__(self, data):
        # data is the user-item interaction matrix
        self.data = data
        self.users = list(set(data.nonzero()[0])) # Unique list of user indices

    def __len__(self):
        # Returns the total number of users
        return len(self.users)

    def __getitem__(self, idx):
        # Retrieves the user index at the specified position
        return torch.tensor(self.users[idx], dtype=torch.long)

class MFModule(nn.Module):
    def __init__(self, num_users, num_items, num_factors=100):
        super().__init__()

        self.num_factors = num_factors
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(num_users, num_factors)  # User embedding
        self.item_embedding = nn.Embedding(num_items, num_factors)  # Item embedding

        # Initialize weights using Xavier initialization
        init.xavier_normal_(self.user_embedding.weight)
        init.xavier_normal_(self.item_embedding.weight)

    def forward(
        self, user_tensor: torch.Tensor, item_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_tensor (torch.Tensor): A tensor containing the indices of the batched users.
            item_tensor (torch.Tensor): A tensor containing the indices of all items.
        --------
        Returns:
            torch.Tensor: A tensor containing the predicted rating scores for each user-item pair, 
                shaped as (number of users in the batch, number of items).
        """
        U_batch = self.user_embedding(user_tensor)
        V = self.item_embedding(item_tensor)

        return U_batch.matmul(V.T)

class FairMF:
    def __init__(self, batch_size=100, max_epochs=250, min_delta=1e-4, learning_rate=1e-3, patience=5, l2=1e-5, num_factors=64, seed=None):
        self.batch_size = batch_size # number of samples to use in each update step
        self.max_epochs = max_epochs # max number of epochs to train
        self.learning_rate = learning_rate # how much to update the weights at each update
        self.patience = patience # number of epochs to wait for an improvement
        self.min_delta = min_delta # a threshold for "significant" change
        self.l2_lambda = l2 # L2 regularization strength
        self.num_factors = num_factors # embedding size
        self.seed = seed # for reproducibility across PyTorch and NumPy

        # If a seed is provided, apply it
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _init_model(self, X:csr_matrix):
        num_users, num_items = X.shape
        self.model_ = MFModule(num_users, num_items, num_factors=self.num_factors).to(self.device)

        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        self.steps = 0 # tracks the number of optimizer steps taken during training

    def fit(self, X: csr_matrix, sst_field: torch.Tensor) -> None:
        """
        Train the model over the input data for a specified number of epochs.
        --------
        Args:
            X (csr_matrix): The user-item interaction matrix, with shape (num_users, num_items).
            sst_field (torch.Tensor): Sensitive Side-information Field.
                A PyTorch tensor of the same shape as X indicating for each user-item pair if it belongs to the protected group. 
                It should be a boolean tensor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model(X)
        self.model_.train()

        self.best_loss = float('inf')
        self.patience_counter = 0

        dataset = InteractionDataset(X)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        item_tensor = torch.arange(X.shape[1]).to(self.device)
        sst_field = sst_field.to(self.device)

        self.epochs = 0  # number of epochs completed
        
        for epoch in range(self.max_epochs):
            losses = []
            for users in data_loader:
                self.optimizer.zero_grad()
                user_tensor = users.to(self.device)
                
                scores = self.model_.forward(user_tensor, item_tensor)
                expected_scores = torch.FloatTensor(X[users].toarray()).to(self.device) # from sparse to tensor - naive
                loss = self._compute_loss(expected_scores, scores, sst_field[users])

                # Backward propagation of the loss
                loss.backward()
                losses.append(loss.item())
                # Update weights according to the gradients
                self.optimizer.step()
                self.steps += 1

            current_loss = np.mean(losses)
            self.epochs += 1

            # Check for improvement
            if self.best_loss - current_loss > self.min_delta:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            print(f"Epoch {epoch+1}/{self.max_epochs}, Loss: {current_loss:.4f}")

            # Early stopping check
            if self.patience_counter >= self.patience:
                break
        
    def _compute_loss(self, true_scores, pred_scores, group_indicator) -> torch.FloatTensor:
        """
        Computes the combined loss which is a sum of the mean squared error (MSE)
        and the L2 regularization term.
        --------
        Args:
            true_scores (torch.Tensor): A 2D tensor containing the true rating scores.
            pred_scores (torch.Tensor): A 2D tensor containing the predicted rating scores.
            group_indicator (torch.Tensor): A 2D boolean tensor indicating the protected group.
        --------
        Returns:
            torch.Tensor: The combined loss value as a scalar tensor.
        """
        mse_loss = F.mse_loss(pred_scores, true_scores, reduction='sum') / true_scores.numel()
        l2_penalty = self.l2_lambda * (self.model_.user_embedding.weight.norm(2)**2 + 
                                       self.model_.item_embedding.weight.norm(2)**2) / 2
        fairness_loss = self._nonparity_unfairness(pred_scores, group_indicator)

        return mse_loss + l2_penalty + fairness_loss
    
    def _nonparity_unfairness(self, pred_scores, group_indicator) -> torch.FloatTensor:
        # Flatten the 2-dimensional tensors
        pred_scores = pred_scores.view(-1); group_indicator = group_indicator.view(-1)

        unique_groups = torch.unique(group_indicator)
        if len(unique_groups) != 2:
            raise ValueError("group_indicator must contain exactly two unique values representing two groups.")
        
        group1 = unique_groups[0]
        group2 = unique_groups[1]
        avg_score_1 = pred_scores[group_indicator == group1].mean()
        avg_score_2 = pred_scores[group_indicator == group2].mean()
        return F.smooth_l1_loss(avg_score_1, avg_score_2)
    
    def predict(self, X: csr_matrix) -> csr_matrix:
        """
        Predicts the rating scores for all users in the given user-item interaction matrix.
        --------
        Args:
            X (csr_matrix): The user-item interaction matrix, with shape (num_users, num_items).
        --------
        Returns:
            csr_matrix: A sparse matrix containing the predicted rating scores.
        """
        self.model_.eval()
        X_pred = lil_matrix(X.shape)
        item_tensor = torch.arange(X.shape[1]).to(self.device)

        dataset = InteractionDataset(X)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():  # No gradient computation for prediction
            for users in data_loader:
                user_tensor = users.to(self.device)
                batch_predictions = self.model_(user_tensor, item_tensor).detach().cpu().numpy()
                X_pred[users.cpu().numpy()] = batch_predictions

        return X_pred.tocsr()