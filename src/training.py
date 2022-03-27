from src.model import RecommenderNet
from src.configs import model_config

import os
import random as rn

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)


def preprocess_df(root_dir, relationships_file, content_file):
    required_columns = ["user_id", "content_id", "rating", "date"]

    df1 = pd.read_csv(os.path.join(root_dir, relationships_file))
    df2 = pd.read_csv(os.path.join(root_dir, content_file))
    joined_df = pd.merge(df1, df2, on="content_id", how="left")
    df = joined_df[required_columns]
    df = df.sort_values("date")
    df = df.astype({"rating": float})
    return df


def create_dataset(root_dir, relationships_file, content_file, validation_split):
    df = preprocess_df(root_dir, relationships_file, content_file)
    user_ids = df["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    content_ids = df["content_id"].unique().tolist()
    content2content_encoded = {x: i for i, x in enumerate(content_ids)}
    content_encoded2content = {i: x for i, x in enumerate(content_ids)}
    df["user"] = df["user_id"].map(user2user_encoded)
    df["content"] = df["content_id"].map(content2content_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(content_encoded2content)
    df["rating"] = df["rating"].values.astype(np.float32)
    # min and max ratings will be used to normalize the ratings later
    min_rating, max_rating = min(df["rating"]), max(df["rating"])

    print(
        f"Number of users: {num_users}, Number of Movies: {num_movies}, Min rating: {min_rating}, Max rating: {max_rating}"
    )

    df = df.sample(frac=1, random_state=1490251)
    x = df[["user", "content"]].values
    y = (
        df["rating"]
        .apply(lambda x: (x - min_rating) / (max_rating - min_rating))
        .values
    )
    train_indices = int(validation_split * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )

    return x_train, x_val, y_train, y_val, num_movies, num_users


if __name__ == "__main__":
    cfg = model_config()

    (
        x_training,
        x_validation,
        y_training,
        y_validation,
        movie_count,
        user_count,
    ) = create_dataset(
        root_dir=cfg.root_dir,
        relationships_file=cfg.relationships_file,
        content_file=cfg.content_file,
        validation_split=cfg.validation_split,
    )
    model = RecommenderNet(user_count, movie_count, 256)

    model.compile(
        loss=tfr.keras.losses.PairwiseHingeLoss(),
        optimizer="adam",
        metrics=[tf.keras.metrics.Recall()],
    )

    callbacks = [
        EarlyStopping(patience=3),
        ReduceLROnPlateau(monitor="val_loss", patience=1),
    ]

    model.fit(
        x=x_training,
        y=y_training,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        verbose=1,
        validation_data=(x_validation, y_validation),
        callbacks=callbacks,
    )

    model.save(cfg.model_checkpoint)
