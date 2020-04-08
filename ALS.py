import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import time


def re_index_column(df, columname):
    unique_values = list(df[columname].unique())

    def process_movie_id(old_id):
        return unique_values.index(old_id)

    df[columname] = df[columname].apply(process_movie_id)

def get_dataframes(num_users, test_percentage):
    rating_df = pd.read_pickle('df.pkl')

    # transform column values
    rating_df['userId'] = rating_df['userId'] -1
    rating_df['rating'] = rating_df['rating']/ 5

    # filter number of users
    rating_df = rating_df.sort_values(by=['userId'])
    rating_df = rating_df[rating_df['userId'] < num_users]

    # filter movies that are not rated among the remaining users
    re_index_column(rating_df, 'movieId')
    num_movies = rating_df['movieId'].nunique()
    print(f'{num_movies} movies left in data')



    train_rating_df, test_rating_df = train_test_split(rating_df, shuffle = True, test_size = test_percentage)



    rating_df = rating_df[['movieId', 'rating', 'userId']]

    per_movieId_df = rating_df[['movieId', 'rating', 'userId']].groupby('movieId').agg(list).reset_index()
    print(per_movieId_df.columns.values.tolist())
    per_userId_df = rating_df[['movieId', 'rating', 'userId']].groupby('userId').agg(list).reset_index()
    print(per_userId_df.columns.values.tolist())

    #umr_train = train_rating_df[['userId','movieId','rating']].values
    umr_test =  test_rating_df[['userId','movieId','rating']].values

    return umr_test, per_userId_df.values, per_movieId_df.values





class Recommender(torch.nn.Module):

    def __init__(self, n_U, n_P, k):
        super(Recommender, self).__init__()

        self.U = torch.nn.Parameter(torch.rand(n_U, k, requires_grad=True))
        self.P = torch.nn.Parameter(torch.rand(n_P, k, requires_grad=True))

    def forward(self, i_user, i_product):
        i_user = int(i_user)
        i_product = int(i_product)
        user_vector = self.U[i_user]
        product_vector = self.P[i_product]
        return user_vector @ product_vector ## inner porduct

    def get_loss(self,umr):
        loss_fn = torch.nn.MSELoss()
        result = 0
        for u,m,r in umr:
            r_hat = self.forward(u, m)
            r = torch.Tensor([r]).squeeze()
            result += loss_fn(r_hat, r)
        return result / len(umr)


def update_P(model, per_movieId_umr):
    for movie, ratings, users in per_movieId_umr:
        indices = torch.LongTensor(users)
        ratings_for_indices = torch.Tensor(ratings).detach().numpy()
        U_tilde = model.U.index_select(0, indices).detach().numpy()
        reg = LinearRegression().fit(U_tilde, ratings_for_indices)
        model.P[movie] = torch.tensor(reg.coef_) #ratings_for_indices  @ U_tilde @ ((U_tilde.T @ U_tilde).inverse())

def update_U(model, per_userId_umr):
    for user, movies, ratings in per_userId_umr:
        indices = torch.LongTensor(movies)
        ratings_for_indices = torch.Tensor(ratings).detach().numpy()
        P_tilde = model.P.index_select(0, indices).detach().numpy()

        reg = LinearRegression().fit(P_tilde, ratings_for_indices)
        model.U[user] = torch.tensor(reg.coef_) #ratings_for_indices  @ P_tilde @ (np.invert(P_tilde.T @ P_tilde))

def main(num_users, test_percentage, N_epochs = 10, k =10):

    umr_test, per_userId_umr, per_movieId_umr = get_dataframes(num_users, test_percentage)
    n_U = len(per_userId_umr)
    print(f'nU: {n_U}')
    n_P = len(per_movieId_umr)
    print(f'nP: {n_P}')

    model = Recommender(n_U, n_P, k)
    print(f"loss_0: {model.get_loss(umr_test)}")

    for ep in range(N_epochs):
        start_ep = time.time()
        print(f'ep: {ep}')
        update_P(model, per_movieId_umr)
        print(f'after updating P loss: {model.get_loss(umr_test)}')
        update_U(model, per_userId_umr)
        print(f'after updating U loss: {model.get_loss(umr_test)}')
        end_ep = time.time() - start_ep
        print(f'this epoch took {end_ep}')












