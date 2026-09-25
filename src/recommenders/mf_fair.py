
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
        
    def _compute_loss(self, true_scores, pred_scores, group_indicator,
                      user_embeddings=None, item_embeddings=None) -> torch.FloatTensor:
        """
        Computes unmasked MSE, L2 regularization, and non-parity loss.
        Optional embeddings allow the same objective to be used for unseen users.
        --------
        Args:
            true_scores (torch.Tensor): A 2D tensor containing the true rating scores.
            pred_scores (torch.Tensor): A 2D tensor containing the predicted rating scores.
            group_indicator (torch.Tensor): A 2D boolean tensor indicating the protected group.
        --------
        Returns:
            torch.Tensor: The combined loss value as a scalar tensor.
        """
        if user_embeddings is None:
            user_embeddings = self.model_.user_embedding.weight
        if item_embeddings is None:
            item_embeddings = self.model_.item_embedding.weight
        mse_loss = F.mse_loss(pred_scores, true_scores, reduction='sum') / true_scores.numel()
        l2_penalty = self.l2_lambda * (user_embeddings.norm(2)**2 +
                                       item_embeddings.norm(2)**2) / 2
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
    
    def predict_new_users(self, X: csr_matrix, sst_field: torch.Tensor) -> csr_matrix:
        """Infer new user factors from input interactions, keeping item factors fixed.

        Use this for strong generalization. Rows of X are local to this call;
        they are never used as IDs in the fitted user embedding table. Columns
        must retain the training item order. sst_field must align with X after
        any row filtering. Only input interactions belong in X, never held-out
        validation/test targets.

        Inference uses the existing unmasked MSE + L2 + non-parity objective
        and the configured learning rate, epoch limit, and stopping settings.
        Zero initialization makes new factors independent of row IDs and RNG
        state. The fitted model and its training statistics are not modified.
        Empty user histories receive zero scores. Seen-item exclusion remains
        the caller's responsibility, as with predict().
        """
        if not hasattr(self, "model_"):
            raise ValueError("Fit FairMF before inferring new users.")
        X = X.tocsr()
        if X.shape[1] != self.model_.num_items:
            raise ValueError("X must have the fitted item columns in training order.")
        if tuple(sst_field.shape) != X.shape:
            raise ValueError("sst_field must have the same shape and row order as X.")

        self.inference_epochs_ = 0
        self.inference_steps_ = 0
        active_users = np.flatnonzero(X.getnnz(axis=1))
        if not len(active_users):
            return csr_matrix(X.shape, dtype=np.float32)

        # Detaching prevents inference gradients from reaching the fitted model.
        item_embeddings = self.model_.item_embedding.weight.detach()
        user_embeddings = nn.Parameter(item_embeddings.new_zeros(
            (X.shape[0], self.num_factors)))
        optimizer = optim.Adam([user_embeddings], lr=self.learning_rate)
        sst_field = sst_field.to(self.device)
        best_loss = float('inf')
        patience_counter = 0

        for _ in range(self.max_epochs):
            losses = []
            for start in range(0, len(active_users), self.batch_size):
                rows = active_users[start:start + self.batch_size]
                users = torch.as_tensor(rows, dtype=torch.long, device=self.device)
                expected_scores = torch.as_tensor(
                    X[rows].toarray(), dtype=item_embeddings.dtype, device=self.device)
                optimizer.zero_grad()
                scores = user_embeddings[users].matmul(item_embeddings.T)
                loss = self._compute_loss(
                    expected_scores, scores, sst_field[users],
                    user_embeddings=user_embeddings, item_embeddings=item_embeddings)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                self.inference_steps_ += 1

            current_loss = np.mean(losses)
            self.inference_epochs_ += 1
            if best_loss - current_loss > self.min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break

        predictions = lil_matrix(X.shape, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(active_users), self.batch_size):
                rows = active_users[start:start + self.batch_size]
                users = torch.as_tensor(rows, dtype=torch.long, device=self.device)
                scores = user_embeddings[users].matmul(item_embeddings.T)
                predictions[rows] = scores.cpu().numpy()
        return predictions.tocsr()

    def predict(self, X: csr_matrix) -> csr_matrix:
        """
        Predicts scores for known users, preserving the original training row IDs.
        For unseen or reindexed users, use predict_new_users() instead.
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
