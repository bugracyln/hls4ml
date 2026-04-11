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
from hgq.layers import QSoftmax
from hgq.regularizers import MonoL1
import hls4ml
from hls4ml.converters import convert_from_keras_model

#k, b, i = 1, 16, 6
k, b, i = 1, 15, 5

scope0 = QuantizerConfigScope(k0=k, b0=b, i0=i, br=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
scope1 = QuantizerConfigScope(place='datalane', k0=k, b0=b, i0=i, fr=MonoL1(1e-8), ir=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
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
    qk_opt = keras.layers.Input((1,EMBEDDING_SIZE_PER_HEAD),batch_size=1, name='qk_opt')

    smax_opt = QSoftmax()(qk_opt)



qmodel = keras.Model(inputs=[qk_opt], outputs=[smax_opt])

qmodel.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', jit_compile=True)

#new_config = {'Model': {'Precision': 'ac_fixed<2,0>', 'ReuseFactor': 1}}
new_config = {'Model': {'Precision': 'ac_fixed<16,6>', 'ReuseFactor': 1}}

model_hls = convert_from_keras_model(qmodel, backend="oneAPI", io_type='io_stream', output_dir='/home/bc/Python/models-hls4ml/streamed_softmax_tests', hls_config=new_config)
model_hls.write()
model_hls.compile()

qk_res = 5 * np.random.uniform(0, 1, (N_TOKENS,1,EMBEDDING_SIZE_PER_HEAD)) - 2.5

keras_pred = qmodel.predict(qk_res).flatten()
hls_pred = model_hls.predict(qk_res).flatten()


print(f'KERAS:\n {keras_pred}')
print(f'HLS:\n {hls_pred}')

np.testing.assert_allclose(hls_pred, keras_pred, rtol=1e-3, atol=1e-3)

print("Success")