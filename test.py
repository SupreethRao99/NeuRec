from tensorflow.keras.models import load_model
from configs import model_config
import tensorflow_ranking as tfr

cfg = model_config()

model = load_model(cfg.model_checkpoint)
print(model.summary())
