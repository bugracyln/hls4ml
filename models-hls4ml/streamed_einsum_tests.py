import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import hgq
import keras
from keras import KerasTensor
from keras.optimizers import AdamW
from keras.layers import Flatten, Concatenate
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.layers import QEinsum, QDense
from hgq.regularizers import MonoL1
import hls4ml
from hls4ml.converters import convert_from_keras_model

scope0 = QuantizerConfigScope(k0=1, b0=16, i0=6, br=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
scope1 = QuantizerConfigScope(place='datalane', k0=1, b0=16, i0=6, fr=MonoL1(1e-8), ir=MonoL1(1e-8), overflow_mode='SAT', round_mode="RND")
betascope = LayerConfigScope(beta0=0.00001)

dim_j = 4 
dim_k = 2
context_len = 4
n_tokens = 10
contract_dim = 0

with betascope, scope0, scope1:

    if not contract_dim:
        input_data = keras.layers.Input((2,1,dim_j,), name='input_data')
        input_kernel = keras.layers.Input((2,1,dim_j,dim_k), name='input_kernel')

        #input_data = keras.layers.Input((1,1, dim_j, 5), name='input_data')
        #input_kernel = keras.layers.Input((1,1, 5,dim_k), name='input_kernel')

        eins_conf = QEinsum(
            equation="...j,...jk->...k",
            name='qeinsum_op',
            context_len=context_len,
            contract_dim=contract_dim
        )

    else:
        input_data = keras.layers.Input((2,1,dim_j,), name='input_data')
        input_kernel = keras.layers.Input((2,1,dim_k,), name='input_kernel')

        assert dim_j == context_len, "For context contraction input vector must have size equal to context length."

        eins_conf = QEinsum(
            equation="...j,...k->...k",
            name='qeinsum_op',
            context_len=context_len,
            contract_dim=1
        )

    output = eins_conf([input_data, input_kernel])


qmodel = keras.Model(inputs=[input_data, input_kernel], outputs=output)
qmodel.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', jit_compile=True)

#new_config = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')
#new_config['Model']['Precision'] = 'ac_fixed<2,0>'
#new_config['Model']['ReuseFactor'] = 1

new_config = {'Model': {'Precision': 'ac_fixed<17,7>', 'ReuseFactor': 1}}

model_hls = convert_from_keras_model(qmodel, backend="oneAPI", io_type='io_stream', output_dir='/home/bc/Python/models-hls4ml/stream_tests', hls_config=new_config)
model_hls.write()
model_hls.compile()

L0 = 1
L1 = 1

expected_opt = np.zeros((context_len))

if not contract_dim:
    data_arr = 5 * np.random.uniform(0, 1, (n_tokens,2,1,dim_j)) - 2.5
    kernel_arr = 5 * np.random.uniform(0, 1, (n_tokens,2,1,dim_j,dim_k)) - 2.5

    #data_arr = 5 * np.random.uniform(0, 1, (n_tokens,1,1, dim_j, 5)) - 2.5
    #kernel_arr = 5 * np.random.uniform(0, 1, (n_tokens,1,1, 5, dim_k)) - 2.5


    if data_arr.shape[-1] != kernel_arr.shape[-1] and len(data_arr.shape) == 4:
        L1 = kernel_arr.shape[-1]
        expected_opt = np.zeros((*kernel_arr.shape[:-2], 1, context_len, kernel_arr.shape[-1]))
    elif data_arr.shape[-1] != kernel_arr.shape[-1] and len(data_arr.shape) == 5:
        L0 = data_arr.shape[-2]
        L1 = kernel_arr.shape[-1]
        expected_opt = np.zeros((*kernel_arr.shape[:-2], L0, context_len, kernel_arr.shape[-1]))
    else:
        expected_opt = np.zeros((*kernel_arr.shape[:-1], 1, context_len,))

    batch_size, seq_size = kernel_arr.shape[1:3]
    for i in range(n_tokens):
        if L0 > 1:
            for l0 in range(L0):
                last_el = (i - context_len + 1) if i >= context_len else 0
                for batch_ptr in range(batch_size):
                    for seq_ptr in range(seq_size):
                        if L1 > 1:
                            for l1 in range(L1): 
                                expected_opt[i][batch_ptr][seq_ptr][l0][:,l1] = np.array([np.dot(data_arr[i][batch_ptr][seq_ptr][l0],kernel_arr[last_el + j][batch_ptr][seq_ptr][:,l1]) if j <= i else 0. for j in range(context_len)])
                        else:
                            expected_opt[i][batch_ptr][seq_ptr][l0] = np.array([np.dot(data_arr[i][batch_ptr][seq_ptr][l0],kernel_arr[last_el + j][batch_ptr][seq_ptr]) if j <= i else 0. for j in range(context_len)])
        else:
            last_el = (i - context_len + 1) if i >= context_len else 0
            for batch_ptr in range(batch_size):
                for seq_ptr in range(seq_size):
                    print(L1)
                    if L1 > 1:
                        for l1 in range(L1): 
                            expected_opt[i][batch_ptr][seq_ptr][0][:,l1] = np.array([np.dot(data_arr[i][batch_ptr][seq_ptr],kernel_arr[last_el + j][batch_ptr][seq_ptr][:,l1]) if j <= i else 0. for j in range(context_len)])
                    else:
                        expected_opt[i][batch_ptr][seq_ptr][0] = np.array([np.dot(data_arr[i][batch_ptr][seq_ptr],kernel_arr[last_el + j][batch_ptr][seq_ptr]) if j <= i else 0. for j in range(context_len)])

else:
    data_arr = 5 * np.random.uniform(0, 1, (n_tokens,2,1,dim_j)) - 2.5
    kernel_arr = 5 * np.random.uniform(0, 1, (n_tokens,2,1,dim_k)) - 2.5

    expected_opt = np.zeros(kernel_arr.shape)
    buffer = np.zeros((context_len,*kernel_arr.shape[1:]))

    batch_size, seq_size = kernel_arr.shape[1:3]
    for i in range(n_tokens):
        if i < context_len:
            buffer[i] =  kernel_arr[i]
        else:
            for ctx in range(context_len-1):
                buffer[ctx] = buffer[ctx+1]
            buffer[-1] = kernel_arr[i]
        
        expected_opt[i] = np.einsum("bsc,cbsk->bsk", data_arr[i], buffer)


expected_opt = expected_opt.flatten()


print(f'DATA ARR:\n {data_arr}')
print(f'KERNEL ARR:\n {kernel_arr}')


keras_pred = qmodel.predict([data_arr, kernel_arr]).flatten()
hls_pred = model_hls.predict([data_arr, kernel_arr]).flatten()

#print(f'KERAS:\n {keras_pred}')
print(f'HLS:\n {hls_pred}')
print(f'EXPECTED OPT:\n {expected_opt}')


np.testing.assert_allclose(hls_pred, expected_opt, rtol=1e-3, atol=1e-3)

print("Success")