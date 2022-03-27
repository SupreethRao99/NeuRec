import tensorflow as tf
from keras import layers
from keras.layers import Dense, Dropout
from tensorflow import keras


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_content, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_content = num_content
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.content_embedding = layers.Embedding(
            num_content,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.content_bias = layers.Embedding(num_content, 1)

        self.d1 = Dense(1024, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(64, activation="relu")
        self.d4 = Dense(1)

        self.dr1 = Dropout(0.3)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        content_vector = self.content_embedding(inputs[:, 1])
        content_bias = self.content_bias(inputs[:, 1])
        dot_user_content = tf.tensordot(user_vector, content_vector, 2)
        # Add all the components (including bias)
        x = dot_user_content + user_bias + content_bias
        x = self.d1(x)
        x = self.dr1(x)
        x = self.d2(x)
        x = self.dr1(x)
        x = self.d3(x)
        x = self.dr1(x)
        x = self.d4(x)

        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)
