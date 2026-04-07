import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import hgq
import keras
from keras.optimizers import AdamW
import keras.ops as ops
from keras.layers import Lambda, Reshape
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.layers import QEinsum, QEinsumDense, QSoftmax, QMultiHeadAttention
from hgq.regularizers import MonoL1
import hls4ml
from hls4ml.converters import convert_from_keras_model
import hls4ml.converters.keras_v3.hgq2.multi_head_attention

scope0 = QuantizerConfigScope(k0=1, b0=16, i0=6, br=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
scope1 = QuantizerConfigScope(place='datalane', k0=1, b0=16, i0=6, fr=MonoL1(1e-8), ir=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
betascope = LayerConfigScope(beta0=0.00001)

#CONSTANTS FOR THE MODEL
EMBEDDING_SIZE = 16
HIDDEN_SIZE = 64
HEADS = 1
CONTEXT_LENGTH = 256
KEY_DIM = HIDDEN_SIZE//HEADS
N_TOKENS = 10


# Token stream
tokens = 5 * np.random.uniform(0, 1, (N_TOKENS,1,1,EMBEDDING_SIZE)) - 2.5

# First generate weights (should take from the model)
scale = 1/np.sqrt(EMBEDDING_SIZE)
W_q = np.random.normal(0, scale, (*tokens.shape[1:], HIDDEN_SIZE))
W_k = np.random.normal(0, scale, (*tokens.shape[1:], HIDDEN_SIZE))
W_v = np.random.normal(0, scale, (*tokens.shape[1:], HIDDEN_SIZE))
W_o = 5 * np.random.uniform(0, 1, (*tokens.shape[1:-1], HIDDEN_SIZE,tokens.shape[-1])) - 2.5


# Buffers to store k and v
kBuffer = np.zeros((CONTEXT_LENGTH, *tokens.shape[1:-1], HIDDEN_SIZE))
vBuffer = np.zeros((CONTEXT_LENGTH, *tokens.shape[1:-1], HIDDEN_SIZE))

att_result = np.zeros(tokens.shape)


for i in range(N_TOKENS):

    # Dense layers to produce qkv vectors
    q_vect = np.einsum("bsc,bsck->bsk",tokens[i], W_q)
    k_vect = np.einsum("bsc,bsck->bsk",tokens[i], W_k)
    v_vect = np.einsum("bsc,bsck->bsk",tokens[i], W_v)

    if i < CONTEXT_LENGTH:
        kBuffer[i] = k_vect
        vBuffer[i] = v_vect
    else:
        for j in range(CONTEXT_LENGTH-1):
            kBuffer[j] = kBuffer[j+1]
            vBuffer[j] = vBuffer[j+1]
        kBuffer[-1] = k_vect
        vBuffer[-1] = v_vect


    # Causal einsum for qK^T calc
    qk_res = np.einsum("bsh,cbsh->bsc",q_vect, kBuffer) / np.sqrt(KEY_DIM)

    # Utilise softmax tables from the actual model

    with betascope, scope0, scope1:
        softmax = QSoftmax()(qk_res)

    sV_res = np.einsum("bsc,cbsh->bsh",softmax, vBuffer)

    # Dense opt layer
    att_result[i] = np.einsum("bsc,bsck->bsk",sV_res, W_o)

    
print(f'Result shape: {att_result.shape} Attention Result:\n {att_result}')

