import os
import json
import tqdm
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from configs import model_config
from training import preprocess_df


def make_recommendations(user_id):
    content = pd.read_csv(os.path.join(cfg.root_dir, cfg.content_file))
    df = preprocess_df(cfg.root_dir, cfg.relationships_file, cfg.content_file)
    user_ids = df["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    content_ids = df["content_id"].unique().tolist()
    content2content_encoded = {x: i for i, x in enumerate(content_ids)}
    content_encoded2content = {i: x for i, x in enumerate(content_ids)}
    df["user"] = df["user_id"].map(user2user_encoded)
    df["content"] = df["content_id"].map(content2content_encoded)
    recs = []
    content_watched_by_user = df[df.user_id == user_id]
    content_not_watched = content[
        ~content["content_id"].isin(content_watched_by_user.content_id.values)
    ]["content_id"]
    content_not_watched = list(
        set(content_not_watched).intersection(set(content2content_encoded.keys()))
    )
    content_not_watched = [
        [content2content_encoded.get(x)] for x in content_not_watched
    ]
    user_encoder = user2user_encoded.get(user_id)
    user_content_array = np.hstack(
        ([[user_encoder]] * len(content_not_watched), content_not_watched)
    )

    model = load_model(cfg.model_checkpoint)
    ratings = model.predict(user_content_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_content_ids = [
        content_encoded2content.get(content_not_watched[x][0])
        for x in top_ratings_indices
    ]
    recommended_content = content[content["content_id"].isin(recommended_content_ids)]
    for row in recommended_content.itertuples():
        recs.append(row.content_id)

    recommendations[user_id] = recs


if __name__ == "__main__":

    cfg = model_config()
    recommendations = {}

    test_df = pd.read_csv(os.path.join(cfg.root_dir, "test.csv"))
    test_list = list(test_df["user_id"])

    for user in tqdm.tqdm(test_list):
        try:
            make_recommendations(user)
        except (ValueError, TypeError):
            pass

    with open("submission.json", "w") as fp:
        json.dump(recommendations, fp)
