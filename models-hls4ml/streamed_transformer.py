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
from hgq.layers import QEinsum, QEinsumDense, QSoftmax
from hgq.regularizers import MonoL1
import hls4ml
from hls4ml.converters import convert_from_keras_model

def layer_weight_set_helper(layer_name, W, B) -> None:
    weights = qmodel.get_layer(layer_name).get_weights()
    weights[0] = W
    weights[1] = B
    qmodel.get_layer(layer_name).set_weights(weights)


scope0 = QuantizerConfigScope(k0=1, b0=16, i0=6, br=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
scope1 = QuantizerConfigScope(place='datalane', k0=1, b0=16, i0=6, fr=MonoL1(1e-8), ir=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
betascope = LayerConfigScope(beta0=0.00001)

#CONSTANTS FOR THE MODEL
EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16
HEADS = 1
CONTEXT_LENGTH = 256
KEY_DIM = HIDDEN_SIZE/HEADS


with betascope, scope0, scope1:

    token = keras.layers.Input((EMBEDDING_SIZE,), name='token')

    one_input = keras.layers.Input((HIDDEN_SIZE,), name='one_in')#for V

    q_vect = QEinsumDense(  equation="...j,kj->...k",
                            output_shape=(HIDDEN_SIZE,),
                            bias_axes="k",
                            name="q_vect",
                            activation="relu")(token)

    #SCALE q HERE WITH 1/SQRT(D_K) FOR qK^T CALC
    #have scaled identity matrix of size hidden x hidden 
    scaled_q_vect = QEinsumDense(  equation="...j,kj->...k",
                            output_shape=(HIDDEN_SIZE,),
                            bias_axes="k",
                            name="scaled_q_vect",
                            activation="relu")(q_vect)
    
    k_vect = QEinsumDense(  equation="...j,kj->...k",
                            output_shape=(HIDDEN_SIZE,1),
                            bias_axes="k",
                            name="k_vect",
                            activation="relu")(token)

    v_vect = QEinsumDense(  equation="...j,kj->...k",
                            output_shape=(HIDDEN_SIZE,1),
                            bias_axes="k",
                            name="v_vect",
                            activation="relu")(token)
    
    v_vect_bram = QEinsum(  equation="...j,...jk->...k",
                            name='v_vect_bram',
                            context_len=CONTEXT_LENGTH,
                            padding_type=0
                            )([one_input, v_vect]) 

    #qk layer infers the output size as 1x1 since IxL1 = 1*1 since it does not know context so we need 
    #to enforce if ctx_len > 1 : out_shape = ctx_len * L1  (not ctx_len * L1 * I since we stream)
    #but keras runs its checks before we enforce this so we need to trick keras somehow
    #this leads to output pipe become 1*1 and activation output is 1*1 so next einsum layer receives 
    #[...,1] and [...,16,1] not 
    
    qk_layer = QEinsum( equation="...j,...jk->...k",
                        name='qk_layer',
                        context_len=CONTEXT_LENGTH,
                        padding_type=1
                        )([scaled_q_vect, k_vect])
    
    activation = QSoftmax()(qk_layer)

    #no context len here we stream 1x1 attention 
    attention_dot = QEinsum(
        equation="...j,...j->...",
        name='attention_dot',
        )([activation, v_vect_bram]) 
    
    '''
    attention = QEinsumDense(   equation="...j,kj->...k",
                                output_shape=(1,),
                                bias_axes="k",
                                name="attention",
                                activation="relu")(attention_dot)
    '''

qmodel = keras.Model(inputs=token, outputs=attention)

#SET WEIGHTS/BIASES HERE
w_q = np.arange(EMBEDDING_SIZE * HIDDEN_SIZE).reshape(HIDDEN_SIZE, EMBEDDING_SIZE)
b_q = np.arange(HIDDEN_SIZE)
layer_weight_set_helper('q_vect', w_q, b_q)

w_scale_q = np.identity(HIDDEN_SIZE)
b_scale_q = np.zeros(HIDDEN_SIZE)
layer_weight_set_helper('scaled_q_vect', w_scale_q, b_scale_q)

w_k = np.arange(EMBEDDING_SIZE * HIDDEN_SIZE).reshape(HIDDEN_SIZE, EMBEDDING_SIZE)
b_k = np.arange(HIDDEN_SIZE)
layer_weight_set_helper('k_vect', w_k, b_k)

w_v = np.arange(EMBEDDING_SIZE * HIDDEN_SIZE).reshape(HIDDEN_SIZE, EMBEDDING_SIZE)
b_v = np.arange(HIDDEN_SIZE)
layer_weight_set_helper('v_vect', w_v, b_v)

#w_ad = np.array(EMBEDDING_SIZE * HIDDEN_SIZE)
#b_ad = np.zeros(HIDDEN_SIZE)
#layer_weight_set_helper('attention', w_ad, b_ad)


qmodel.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', jit_compile=True)

new_config = {'Model': {'Precision': 'ac_fixed<2,0>', 'ReuseFactor': 1}}

model_hls = convert_from_keras_model(qmodel, backend="oneAPI", io_type='io_stream', output_dir='/home/bc/Python/models-hls4ml/streamed_transformer_tests', hls_config=new_config)
model_hls.write()
model_hls.compile()

#INPUTS
tokens = 5 * np.random.uniform(0, 1, (10, EMBEDDING_SIZE)) - 2.5
identity_matrix_stream = np.eye(HIDDEN_SIZE)

#USE MODELS
keras_pred = qmodel.predict(tokens).flatten()
hls_pred = model_hls.predict(tokens).flatten()

np.testing.assert_allclose(keras_pred, hls_pred, rtol=1e-2, atol=1e-2)

print("Success")