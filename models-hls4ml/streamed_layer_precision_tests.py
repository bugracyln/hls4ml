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
    token_q_dummy = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='token_q_dummy') # no batching and singular vector hence (1, 1, 1, EMBEDDING_SIZE_PER_HEAD)
    token_k_dummy = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='token_k_dummy')
    token_v_dummy = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='token_v_dummy')
    attention_conf_dummy = QMultiHeadAttention(num_heads=HEADS, key_dim=KEY_DIM, value_dim=KEY_DIM, context_len=CONTEXT_LENGTH)
    attention_dummy = attention_conf_dummy(token_q_dummy, token_k_dummy, token_v_dummy)

with betascope, scope0, scope1:
    token_q = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='token_q') # no batching and singular vector hence (1, 1, 1, EMBEDDING_SIZE_PER_HEAD)
    token_k = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='token_k')

    config_to_Q = attention_conf_dummy._layers[0]

    config_to_K = attention_conf_dummy._layers[1]
    config_to_K.name = 'layer_key'

    #config_to_V = QEinsumDense.from_config(attention_conf_dummy._layers[2].get_config())
    config_einsum_QK = QEinsum("abcj,abcj->acbk", name='att_QK', context_len=CONTEXT_LENGTH, contract_dim=0)

    query, key = config_to_Q(token_q), config_to_K(token_k)
    qK = config_einsum_QK([query, key])


qmodel = keras.Model(inputs=[token_q, token_k], outputs=[qK]) 

qmodel.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', jit_compile=True)

#new_config = {'Model': {'Precision': 'ac_fixed<2,0>', 'ReuseFactor': 1}}
new_config = {'Model': {'Precision': 'ac_fixed<16,6>', 'ReuseFactor': 1}}

model_hls = convert_from_keras_model(qmodel, backend="oneAPI", io_type='io_stream', output_dir='/home/bc/Python/models-hls4ml/streamed_layer_precision_tests', hls_config=new_config)
model_hls.write()
model_hls.compile()

tokens = 5 * np.random.uniform(0, 1, (N_TOKENS,1,EMBEDDING_SIZE_PER_HEAD)) - 2.5

keras_pred = qmodel.predict([tokens,tokens]).flatten() 
hls_pred = model_hls.predict([tokens,tokens]).flatten()


W_q = attention_conf_dummy._layers[0].get_weights()[0] # embedding, 1, hidden
W_k = attention_conf_dummy._layers[1].get_weights()[0] # embedding, 1, hidden
W_v = attention_conf_dummy._layers[2].get_weights()[0] # embedding, 1, hidden
W_o = attention_conf_dummy._layers[5].get_weights()[0] # 1, hidden, embedding

# Buffers to store k and v
kBuffer = np.zeros((CONTEXT_LENGTH, *tokens.shape[1:-1], HIDDEN_SIZE))
vBuffer = np.zeros((CONTEXT_LENGTH, *tokens.shape[1:-1], HIDDEN_SIZE))

att_result = np.zeros(tokens.shape)

res = []
for i in range(N_TOKENS):

    # Dense layers to produce qkv vectors
    q_vect = np.einsum("bc,cbh->bh",tokens[i], W_q)
    k_vect = np.einsum("bc,cbh->bh",tokens[i], W_k)
    #v_vect = np.einsum("bc,cbh->bh",tokens[i], W_v)
    
    if i < CONTEXT_LENGTH:
        kBuffer[i] = k_vect
        #vBuffer[i] = v_vect
    else:
        for j in range(CONTEXT_LENGTH-1):
            kBuffer[j] = kBuffer[j+1]
            vBuffer[j] = vBuffer[j+1]
        kBuffer[-1] = k_vect
        #vBuffer[-1] = v_vect

    # Causal einsum for qK^T calc
    qk_res = np.einsum("bh,cbh->bc",q_vect, kBuffer) #/ np.sqrt(KEY_DIM)
    res.append(qk_res)

    # Utilise softmax tables from the actual model

    #with betascope, scope0, scope1:
    #    softmax = QSoftmax()(qk_res)

    #sV_res = np.einsum("bc,cbh->bh",softmax, vBuffer)

    # Dense opt layer
    #att_result[i] = np.einsum("bc,bck->bk",sV_res, W_o)

#att_result = att_result.flatten()    
#print(f'Result shape: {att_result.shape} Keras Attention Result:\n {att_result}')
res = np.array(res).flatten()
print(f'Result shape: {res.shape} Keras qK Result:\n {np.array([res[256*i:256*i+10] for i in range(10)]).flatten()}')


#print(f'KERAS:\n {keras_pred}')
print(f'HLS:\n {np.array([hls_pred[256*i:256*i+10] for i in range(10)]).flatten()}')

#np.testing.assert_allclose(hls_pred, res, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(np.array([hls_pred[256*i:256*i+10] for i in range(10)]).flatten(), np.array([res[256*i:256*i+10] for i in range(10)]).flatten(), rtol=1e-3, atol=1e-3)

print("Success")