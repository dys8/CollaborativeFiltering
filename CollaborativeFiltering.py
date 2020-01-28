# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 01:37:05 2019

@author: duygu
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Loading the movies dataset
movies_df = pd.read_csv('./data/ml-latest/movies.csv', sep=',', header=None, engine='python')

#Loading the ratings dataset
ratings_df = pd.read_csv('./data/ml-latest/ratings.csv', sep=',', header=None, engine='python')


movies_df.columns = ['MovieID', 'Title', 'Genres']
#print(movies_df.head())
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']


user_rating_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')
norm_user_rating_df = user_rating_df.fillna(0) / 5.0 #normalizing user ratings

trX = norm_user_rating_df.values


hiddenUnits = 15
visibleUnits =  len(user_rating_df.columns)

vb = tf.placeholder("float", [visibleUnits]) #number of movies
hb = tf.placeholder("float", [hiddenUnits]) #number of latent features 
W = tf.placeholder("float", [visibleUnits, hiddenUnits])


#Processing the input
v0 = tf.placeholder("float", [None, visibleUnits])
h_0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(h_0 - tf.random_uniform(tf.shape(h_0))))

#Reconstruction of input
v_1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) 
v1 = tf.nn.relu(tf.sign(v_1 - tf.random_uniform(tf.shape(v_1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

alpha = 1.0 #learning rate

#Creating the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

#Calculating the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

#Creating update operations for the weights and biases
w_updated = W + alpha * CD
vb_updated = vb + alpha * tf.reduce_mean(v0 - v1, 0)
hb_updated = hb + alpha * tf.reduce_mean(h0 - h1, 0)

#error
err = v0 - v1
err_sum = tf.reduce_mean(err * err)

#current weights
w_c = np.zeros([visibleUnits, hiddenUnits], np.float32)
#current visible unit biases
vb_c = np.zeros([visibleUnits], np.float32)
#current hidden unit biases
hb_c = np.zeros([hiddenUnits], np.float32)
#previous weights
w_p = np.zeros([visibleUnits, hiddenUnits], np.float32)
#previous visible unit biases
vb_p = np.zeros([visibleUnits], np.float32)
#previous hidden unit biases
hb_p = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 25
batchsize = 60
errors = []
for i in range(epochs):
    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        w_c = sess.run(w_updated, feed_dict={v0: batch, W: w_p, vb: vb_p, hb: hb_p})
        vb_c = sess.run(vb_updated, feed_dict={v0: batch, W: w_p, vb: vb_p, hb: hb_p})
        nb_c = sess.run(hb_updated, feed_dict={v0: batch, W: w_p, vb: vb_p, hb: hb_p})
        w_p = w_c
        vb_p = vb_c
        hb_p = hb_c
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: w_c, vb: vb_c, hb: hb_c}))
    print (errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()

mock_user_id = 186

#selecting an input user
inputUser = trX[mock_user_id-1].reshape(1, -1)
#print(inputUser[0:5])

#feeding in the user and reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={ v0: inputUser, W: w_p, hb: hb_p})
rec = sess.run(vv1, feed_dict={ hh0: feed, W: w_p, vb: vb_p})

print(rec)

scored_movies_df_mock = movies_df[movies_df['MovieID'].isin(user_rating_df.columns)]
scored_movies_df_mock = scored_movies_df_mock.assign(RecommendationScore = rec[0])
scored_movies_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20)

movies_df_mock = ratings_df[ratings_df['UserID'] == mock_user_id]
movies_df_mock.head()

#merging movies_df with ratings_df by MovieID
merged_df_mock = scored_movies_df_mock.merge(movies_df_mock, on='MovieID', how='outer')
print(merged_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20))

