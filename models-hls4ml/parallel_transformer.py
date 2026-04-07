import numpy as np
import tensorflow as tf
from tensorflow import keras
from qkeras import QDense, QActivation
import hls4ml

# -----------------------------
# Model dimensions (TINY)
# -----------------------------
D = 4  # embedding dim

# -----------------------------
# Inputs (FLAT!)
# -----------------------------
x = keras.Input(shape=(D,), name="token")

# -----------------------------
# Q, K, V projections
# -----------------------------
Q = QDense(
    D,
    kernel_quantizer="quantized_bits(8,0,1)",
    bias_quantizer="quantized_bits(8,0,1)",
    name="q_dense",
)(x)

K = QDense(
    D,
    kernel_quantizer="quantized_bits(8,0,1)",
    bias_quantizer="quantized_bits(8,0,1)",
    name="k_dense",
)(x)

V = QDense(
    D,
    kernel_quantizer="quantized_bits(8,0,1)",
    bias_quantizer="quantized_bits(8,0,1)",
    name="v_dense",
)(x)

# -----------------------------
# Attention score (dot product)
# -----------------------------
score = keras.layers.Dot(axes=1, name="qk_dot")([Q, K])

# -----------------------------
# Softmax over scalar (toy!)
# -----------------------------
alpha = keras.layers.Activation("softmax", name="attn_softmax")(score)

# -----------------------------
# Apply attention
# -----------------------------
out = keras.layers.Multiply(name="attn_apply")([alpha, V])

model = keras.Model(x, out)
model.summary()


config = hls4ml.utils.config_from_keras_model(
    model, granularity="name"
)

config["Model"]["Precision"] = "ap_fixed<16,6>"
config["Model"]["ReuseFactor"] = 1

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    backend="oneAPI",
    io_type="io_stream",
    hls_config=config,
    output_dir="qkeras_attention_parallel",
)

hls_model.compile()