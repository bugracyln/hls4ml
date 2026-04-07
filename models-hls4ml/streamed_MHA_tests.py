import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import hgq
import keras
from keras.optimizers import AdamW
import keras.ops as ops
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.layers import QSoftmax, QMultiHeadAttention
from hgq.regularizers import MonoL1
import hls4ml
from hls4ml.converters import convert_from_keras_model
import hls4ml.converters.keras_v3.hgq2.multi_head_attention
from fxpmath import Fxp

#s0_k, s0_b, s0_i = 1, 16, 6
#s1_k, s1_b, s1_i = 1, 16, 6
s0_k, s0_b, s0_i = 1, 16, 6
s1_k, s1_b, s1_i = 1, 16, 6

scope0 = QuantizerConfigScope(k0=s0_k, b0=s0_b, i0=s0_i, br=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
scope1 = QuantizerConfigScope(place='datalane', k0=s1_k, b0=s1_b, i0=s1_i, fr=MonoL1(1e-8), ir=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
betascope = LayerConfigScope(beta0=0.00001)

#CONSTANTS FOR THE MODEL
#NOTE: INTERNALLY VECTOR IS CONVERTED FROM (HIDDEN_SIZE,) TO (HEADS, HIDDEN_SIZE//HEADS)
EMBEDDING_SIZE_PER_HEAD = 16
HIDDEN_SIZE = 64
HEADS = 1 #DO NOT USE STREAM HEAD BY HEAD
CONTEXT_LENGTH = 256
KEY_DIM = HIDDEN_SIZE//HEADS
N_TOKENS = 10


with betascope, scope0, scope1:
    token_q = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='token_q') # no batching and singular vector hence (1, 1, 1, EMBEDDING_SIZE_PER_HEAD)
    token_k = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='token_k')
    token_v = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='token_v')

    # KEEP HEADS AT 1 BUT STREAM IN HEAD SEPERATED ORDER !!!!!!!
    attention_conf = QMultiHeadAttention(num_heads=HEADS, key_dim=KEY_DIM, value_dim=KEY_DIM, context_len=CONTEXT_LENGTH, dtype='float16')

    attention = attention_conf(token_q, token_k, token_v)
    

qmodel = keras.Model(inputs=[token_q, token_k, token_v], outputs=[attention])

qmodel.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', jit_compile=True)

#new_config = {'Model': {'Precision': 'ac_fixed<2,0>', 'ReuseFactor': 1}}
new_config = {'Model': {'Precision': 'ac_fixed<17,7>', 'ReuseFactor': 1}}

model_hls = convert_from_keras_model(qmodel, backend="oneAPI", io_type='io_stream', output_dir='/home/bc/Python/models-hls4ml/streamed_MHA_tests', hls_config=new_config)
model_hls.write()
model_hls.compile()


tokens = 5 * np.random.uniform(0, 1, (N_TOKENS,1,EMBEDDING_SIZE_PER_HEAD)) - 2.5


keras_pred = qmodel.predict([tokens,tokens,tokens]).flatten()
hls_pred = model_hls.predict([tokens,tokens,tokens]).flatten()


# Keras does not work with causal einsum so we create our custom MHA based on the inputs.
# First generate weights (should take from the model)
#scale = 1/np.sqrt(EMBEDDING_SIZE_PER_HEAD)
#W_q = np.random.normal(0, scale, (*tokens.shape[1:], HIDDEN_SIZE))
#W_k = np.random.normal(0, scale, (*tokens.shape[1:], HIDDEN_SIZE))
#W_v = np.random.normal(0, scale, (*tokens.shape[1:], HIDDEN_SIZE))
#W_o = 5 * np.random.uniform(0, 1, (*tokens.shape[1:-1], HIDDEN_SIZE,tokens.shape[-1])) - 2.5




def fxp(x):
    return Fxp(x, signed=True, n_word=16, n_frac=10).get_val()

W_q = attention_conf._layers[0].get_weights()[0] # embedding, 1, hidden
W_k = attention_conf._layers[1].get_weights()[0] # embedding, 1, hidden
W_v = attention_conf._layers[2].get_weights()[0] # embedding, 1, hidden
W_o = attention_conf._layers[5].get_weights()[0] # 1, hidden, embedding

# Buffers to store k and v
kBuffer = np.zeros((CONTEXT_LENGTH, *tokens.shape[1:-1], HIDDEN_SIZE))
vBuffer = np.zeros((CONTEXT_LENGTH, *tokens.shape[1:-1], HIDDEN_SIZE))

att_result = np.zeros(tokens.shape)


for i in range(N_TOKENS):

    # Dense layers to produce qkv vectors
    q_vect = np.einsum("bc,cbh->bh",tokens[i], W_q)
    k_vect = np.einsum("bc,cbh->bh",tokens[i], W_k)
    v_vect = np.einsum("bc,cbh->bh",tokens[i], W_v)
    
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
    qk_res = np.einsum("bh,cbh->bc",q_vect, kBuffer) #/ np.sqrt(KEY_DIM)

    # Utilise softmax tables from the actual model

    with betascope, scope0, scope1:
        softmax = QSoftmax()(qk_res)

    sV_res = np.einsum("bc,cbh->bh",softmax, vBuffer)

    # Dense opt layer
    att_result[i] = np.einsum("bc,bck->bk",sV_res, W_o)

att_result = att_result.flatten()    
print(f'Result shape: {att_result.shape} Keras Attention Result:\n {att_result}')


#print(f'KERAS:\n {keras_pred}')
print(f'HLS:\n {hls_pred}')

np.testing.assert_allclose(hls_pred, att_result, rtol=1e-3, atol=1e-3)

print("Success")