import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import hgq
import keras
from keras.optimizers import AdamW
from keras.layers import Flatten, Concatenate
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.layers import QEinsum, QDense, QEinsumDense
from hgq.regularizers import MonoL1
import hls4ml
from hls4ml.converters import convert_from_keras_model

scope0 = QuantizerConfigScope(k0=1, b0=16, i0=6, br=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
scope1 = QuantizerConfigScope(place='datalane', k0=1, b0=16, i0=6, fr=MonoL1(1e-8), ir=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
betascope = LayerConfigScope(beta0=0.00001)

dim_j = 6 
dim_k = 4 

with betascope, scope0, scope1:
    '''
    input_data = keras.layers.Input((dim_j, dim_k), name='input_data')

    output = QDense(dim_k, activation='relu')(input_data)
    '''
    # this can be added when we have an einsum stream layer to convert
    # probably requires fixing the dimensions since this was just used for testing

    input_data = keras.layers.Input((dim_j,), name='input_data')
    #input_kernel = keras.layers.Input((dim_j, dim_k), name='input_kernel')

    output_conf = QEinsumDense(
        equation="...j,kj->...k",
        output_shape=(dim_k,),
        bias_axes="k",
        name="qeinsum_op",
        activation="relu", 
        dtype="float32"
    )

    output = output_conf(input_data)

import pdb; pdb.set_trace()
qmodel = keras.Model(inputs=input_data, outputs=output)
W = np.arange(dim_j * dim_k).reshape(dim_k, dim_j)
B = np.arange(dim_k, dtype="float32")


#for w in qmodel.get_layer("qeinsum_op").weights:
#    print(w.name, w.shape)

weights = qmodel.get_layer('qeinsum_op').get_weights()
weights[0] = W
#weights[1] = B
qmodel.get_layer('qeinsum_op').set_weights(weights)

qmodel.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', jit_compile=True)

#new_config = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')
#new_config['Model']['Precision'] = 'ac_fixed<2,0>'
#new_config['Model']['ReuseFactor'] = 1

new_config = {'Model': {'Precision': 'ac_fixed<2,0>', 'ReuseFactor': 1}}

model_hls = convert_from_keras_model(qmodel, backend="oneAPI", io_type='io_stream', output_dir='/home/bc/Python/models-hls4ml/dense_stream_tests', hls_config=new_config)
model_hls.write()
model_hls.compile()

data_arr = 5 * np.random.uniform(0, 1, (10, dim_j)) - 2.5

keras_pred = qmodel.predict(data_arr).flatten()
#print(f'KERAS: {keras_pred}')

hls_pred = model_hls.predict(data_arr).flatten()
#print(f'HLS: {hls_pred}')

np.testing.assert_allclose(keras_pred, hls_pred, rtol=1e-3, atol=1e-3)

print("Success")