
import random
import time

import numpy as np
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# FDA (Fair Data Augmentation) with BPR (Bayesian Personalized Ranking) as its backbone model
# From the paper "Improving Recommendation Fairness via Data Augmentation" presented at WWW 2023 Conference
# This implementation is a reorganized version of the original code from https://github.com/newlei/FDA
# All credits go to the authors of the original code

class BPRData(Dataset):
    def __init__(self, train_dict=None, num_items=0, num_ng=5, data_set_count=0, seed=None):
        super(BPRData, self).__init__()

        self.num_items = num_items
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.data_set_count = data_set_count 
        self.set_all_item = set(range(num_items))
        self.features_fill = []
        self.seed = seed

    def ng_sample(self):
        np.random.seed(self.seed)
        self.seed += 1 # ensures different negative samples

        for user_id in self.train_dict:
            positive_list = self.train_dict[user_id]

            for item_i in positive_list:
                # item_i: positive item, item_j: negative item
                for _ in range(self.num_ng):
                    item_j = np.random.randint(self.num_items) 
                    while item_j in positive_list:
                        item_j = np.random.randint(self.num_items)
                    self.features_fill.append([user_id, item_i, item_j]) 

    def __len__(self):  
        return len(self.features_fill)

    def __getitem__(self, idx):
        features = self.features_fill  
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return user, item_i, item_j

class FairData(nn.Module):
    def __init__(self, num_users, num_items, num_factors, noise_ratio, users_features):
        super(FairData, self).__init__()
        """
        num_users: number of users;
        num_items: number of items;
        num_factors: number of predictive factors;
        noise_ratio: ratio of maximum fake data and number of real items;
        users_features: numpy ndarray binary indicating non-protected group (1 for non-protected).
        """ 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.users_features = torch.LongTensor(users_features).to(self.device) 
        self.num_users = num_users
        self.num_factors = num_factors
        self.noise_ratio = noise_ratio
        
        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors) 
        
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        
        self.noise_item = nn.Embedding(num_items, num_factors)    
        nn.init.normal_(self.noise_item.weight, std=0.01)
        
        self.min_clamp=-1
        self.max_clamp=1
   
    def fake_pos(self, male_noise_i_emb, female_noise_i_emb): 
        male_len = male_noise_i_emb.shape[0]
        female_len = female_noise_i_emb.shape[0]    
 
        avg_len = 1
        male_end_idx = male_len%avg_len+avg_len
        male_noise_i_reshape = male_noise_i_emb[:-male_end_idx].reshape(-1,avg_len, self.num_factors)
        male_noise_i_mean = torch.mean(male_noise_i_reshape,axis=1)
        male_noise_len = male_noise_i_mean.shape[0]
        if male_noise_len > female_len:
            female_like = male_noise_i_mean[:female_len]
        else:
            expand_len = int(female_len/male_noise_len)+1 
            female_like = male_noise_i_mean.repeat(expand_len,1)[:female_len]
        
        female_end_idx = female_len%avg_len+avg_len 
        female_noise_i_emb_reshape = female_noise_i_emb[:-female_end_idx].reshape(-1,avg_len,self.num_factors)
        female_noise_i_mean = torch.mean(female_noise_i_emb_reshape,axis=1)
        female_noise_len = female_noise_i_mean.shape[0]
        if female_noise_len > male_len:
            male_like = female_noise_i_mean[:male_len]
        else:
            expand_len = int(male_len/female_noise_len)+1
            male_like = female_noise_i_mean.repeat(expand_len,1)[:male_len]
            
        return male_like, female_like
        
    def forward(self, u_batch, i_batch, j_batch):
        # filter gender
        user_emb = self.embed_user.weight 
        item_emb = self.embed_item.weight  
        noise_emb =  self.embed_item.weight 
        noise_emb = torch.clamp(noise_emb, min=self.min_clamp, max=self.max_clamp) 
        noise_emb += self.noise_item.weight
        
        #get gender attribute
        gender = F.embedding(u_batch, torch.unsqueeze(self.users_features, 1)).reshape(-1)
        male_gender = gender.type(torch.BoolTensor).to(self.device)
        female_gender = (1 - gender).type(torch.BoolTensor).to(self.device)
        
        u_emb = F.embedding(u_batch,user_emb)
        i_emb = F.embedding(i_batch,item_emb)  
        j_emb = F.embedding(j_batch,item_emb) 
        noise_i_emb2 = F.embedding(i_batch,noise_emb)
        len_noise = int(i_emb.size()[0]*self.noise_ratio)
        add_emb = torch.cat((i_emb[:-len_noise],noise_i_emb2[-len_noise:]),0)

        noise_j_emb2 = F.embedding(j_batch,noise_emb)
        len_noise = int(j_emb.size()[0]*self.noise_ratio)
        add_emb_j = torch.cat((noise_j_emb2[-len_noise:],j_emb[:-len_noise]),0)
        
        #according gender attribute, selecting embebdding
        male_i_batch = torch.masked_select(i_batch, male_gender)
        female_i_batch = torch.masked_select(i_batch, female_gender) 
        male_noise_i_emb = F.embedding(male_i_batch,noise_emb) 
        female_noise_i_emb = F.embedding(female_i_batch,noise_emb)   
        male_like_emb, female_like_emb = self.fake_pos(male_noise_i_emb,female_noise_i_emb)
        
        male_j_batch = torch.masked_select(j_batch, male_gender)
        female_j_batch = torch.masked_select(j_batch, female_gender) 
        male_j_emb = F.embedding(male_j_batch,item_emb) 
        female_j_emb = F.embedding(female_j_batch,item_emb)  
        
        male_u_batch = torch.masked_select(u_batch, male_gender)
        female_u_batch = torch.masked_select(u_batch, female_gender) 
        male_u_emb = F.embedding(male_u_batch,user_emb) 
        female_u_emb = F.embedding(female_u_batch,user_emb)  
        
        prediction_neg = (u_emb * add_emb_j).sum(dim=-1) 
        prediction_add = (u_emb * add_emb).sum(dim=-1)
        loss_add = -((prediction_add - prediction_neg).sigmoid().log().mean()) 
        l2_regulization = 0.01*(u_emb**2+add_emb**2+j_emb**2).sum(dim=-1).mean() 
        
        prediction_neg_male = (male_u_emb * male_j_emb).sum(dim=-1) 
        prediction_pos_male = (male_u_emb * male_like_emb).sum(dim=-1)
        loss_fake_male = -((prediction_pos_male - prediction_neg_male).sigmoid().log().mean()) 
        prediction_neg_female = (female_u_emb * female_j_emb).sum(dim=-1) 
        prediction_pos_female = (female_u_emb * female_like_emb).sum(dim=-1)
        loss_fake_female = -((prediction_pos_female - prediction_neg_female).sigmoid().log().mean()) 
        loss_fake = loss_fake_male + loss_fake_female
        l2_regulization2 = 0.01*(male_like_emb**2).sum(dim=-1).mean()+ 0.01*(female_like_emb**2).sum(dim=-1).mean() 
        
        loss_task = 1*loss_add + l2_regulization
        loss_add_item = loss_fake + l2_regulization2
        all_loss = [loss_task, l2_regulization, loss_add_item]

        return all_loss
    
    def predict(self):
        """
        Returns the product of user and item embedding matrices as a CSR matrix.
        """
        u_emb = self.embed_user.weight.detach()  # User embeddings detached from the graph
        i_emb = self.embed_item.weight.detach()  # Item embeddings detached from the graph

        # Compute the dot product
        product = torch.mm(u_emb, i_emb.T).cpu().numpy()

        return csr_matrix(product)
    
class FDA_bpr:
    def __init__(self, batch_size=2048*100, max_epochs=135, num_factors=64, num_ng=5, noise_ratio=0.4, seed=None):
        # all params are the default from the authors' original code
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_factors = num_factors
        self.num_ng = num_ng
        self.noise_ratio = noise_ratio # determines the number of fake interactions involved during training
        self.seed = seed

        # If a seed is provided, apply it
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _init_model(self, X, users_features):
        num_users, num_items = X.shape
        self.model_ = FairData(num_users, num_items, self.num_factors, self.noise_ratio, users_features).to(self.device)

        # init optimizers
        self.task_optimizer = torch.optim.Adam(list(self.model_.embed_user.parameters()) + \
                                    list(self.model_.embed_item.parameters()),lr=1e-3)
        self.noise_optimizer = torch.optim.Adam(list(self.model_.noise_item.parameters()),lr=1e-3)

    def fit(self, X, users_features) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model(X, users_features)
        self.model_.train() # Set the model to training mode

        num_items = X.shape[1] # Number of items (columns in X)
        num_interactions = X.nnz # Number of interactions (non-zero entries in X)
        train_dict = {user_id: X[user_id].nonzero()[1].tolist() for user_id in range(X.shape[0])}

        train_dataset = BPRData(train_dict=train_dict, num_items=num_items, num_ng=self.num_ng, data_set_count=num_interactions, seed=self.seed)
        start_time = time.time()
        print('Negative sampling, start')
        train_dataset.ng_sample()
        print('Negative sampling, end')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.max_epochs):
            loss_current = [[], [], []]

            # First optimization loop for task_loss
            for user_batch, itemi_batch, itemj_batch in train_loader:
                user_batch = user_batch.to(self.device)
                itemi_batch = itemi_batch.to(self.device)
                itemj_batch = itemj_batch.to(self.device)

                # Forward pass
                get_loss = self.model_.forward(user_batch, itemi_batch, itemj_batch)
                task_loss, relu_loss, noise_loss = get_loss
                loss_current[0].append(task_loss.item())
                loss_current[1].append(relu_loss.item())

                self.task_optimizer.zero_grad()
                task_loss.backward()
                self.task_optimizer.step()

            # Second optimization loop for noise_loss
            for user_batch, itemi_batch, itemj_batch in train_loader:
                user_batch = user_batch.to(self.device)
                itemi_batch = itemi_batch.to(self.device)
                itemj_batch = itemj_batch.to(self.device)

                # Forward pass
                get_loss = self.model_.forward(user_batch, itemi_batch, itemj_batch)
                task_loss, relu_loss, noise_loss = get_loss
                loss_current[2].append(noise_loss.item())

                self.noise_optimizer.zero_grad()
                noise_loss.backward()
                self.noise_optimizer.step()

            # Loss computation and logging
            loss_current = np.array(loss_current)
            elapsed_time = time.time() - start_time
            train_loss_task = round(np.mean(loss_current[0]), 3)
            # train_loss_sample = round(np.mean(loss_current[1]), 3)
            # train_loss_noise = round(np.mean(loss_current[2]), 3)
            str_print_train = f"epoch:{epoch + 1} time:{round(elapsed_time, 1)} loss task:{train_loss_task}"
            print(str_print_train)
