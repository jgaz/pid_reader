import tensorflow as tf
from official.vision.image_classification.efficientnet import efficientnet_model
import sys

backbone_model_path = "/home/jgaz/code/pid_reader/trainer/src/https/storageaccountdatav9498.blob.core.windows.net/pub/21dc09821e6e4b722b93878a078977483ba798dd/backbone"
model = tf.keras.models.load_model(backbone_model_path)
with open("./backbone-custom.log", "w") as fout:
    sys.stdout = fout
    model.summary()

model = efficientnet_model.EfficientNet.from_name("efficientnet-b0")
with open("./backbone-official.log", "w") as fout:
    sys.stdout = fout
    model.summary()
