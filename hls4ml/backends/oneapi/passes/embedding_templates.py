"""
These are the stream oneAPI templates for embedding layers. The io_parallel ones are in backends/fpga/passes/embedding.py.
"""

import shutil
from hls4ml.backends.fpga.fpga_backend import Embedding
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Embedding

embed_config_template = """struct config{index} : nnet::embed_config {{
    static constexpr unsigned n_in = {n_in};
    static constexpr unsigned n_out = {n_out};
    static constexpr unsigned vocab_size = {vocab_size};
    static constexpr unsigned io_type = nnet::{iotype};
    static constexpr unsigned reuse_factor = {reuse};
    static constexpr bool embed_pos = {embed_pos};
    static constexpr unsigned num_banks = DIV_ROUNDUP(n_out, reuse_factor);
    typedef {embeddings_t.name} embeddings_t;
    [[intel::fpga_memory, intel::numbanks(num_banks),
    intel::bankwidth(sizeof(embeddings_t::value_type))]] static constexpr embeddings_t embeddings = {e};
}};\n"""


embed_function_template = 'nnet::embedding<{input_t}, {output_t}, {config}>({input}, {output});'

embed_include_list = ['nnet_utils/nnet_embed.h', 'nnet_utils/nnet_embed_stream.h']

embed_task_sequence_template = 'task_sequence<nnet::embedding_stream<{input_pipe}, {output_pipe}, {config}>> {name};'

embed_task_sequence_template_max_invoc = (
    'task_sequence<nnet::embedding_stream<{input_pipe}, {output_pipe}, {config}>,{maxinvoc}>  {name};'
)

embed_stream_function_template = '{name}.async();'


class EmbeddingConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Embedding)
        self.template = embed_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['e'] = node.get_weights('embeddings').name

        autoreg_model: bool = node.model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None
        params['embed_pos'] = 'false'

        if autoreg_model:
            embed_pos = node.model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault('InpPosStream', None)

            if embed_pos is not None:
                assert embed_pos in [layer.name for layer in node.model.get_layers() if isinstance(layer, Embedding)], (
                    "The positional embedding layer name is not found in the model, "
                    "make sure layer names match and layer type is embedding. "
                    "Specify the name as: 'InpPosStream' : '<embedding_layer_name>'")
            else:
                print("Warning: Positinal embedding not specified, positional embedding will not be used in this model")
            
            if (node.name == embed_pos):
                params['embed_pos'] = 'true' 

        return self.template.format(**params)


class EmbeddingFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Embedding, include_header=embed_include_list)
        self.template = embed_function_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)


class EmbeddingTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Embedding)
        self.template = embed_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)

        max_invoc = node.model.config.get_config_value('HLSConfig').setdefault('MaxInvoc', None)
        if max_invoc is not None:
            self.template = embed_task_sequence_template_max_invoc
            params['maxinvoc'] = 'ts_invoc_props' if shutil.which('ahls') else f'{max_invoc},{max_invoc}'
        
        autoreg_model: bool = node.model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None

        if autoreg_model:
            model_inp_names = [layer.pipe_name for layer in node.model.get_input_variables()] 
            model_out_names = [layer.pipe_name for layer in node.model.get_output_variables()]

            if (params['input_pipe'] in model_inp_names):
                params['input_pipe'] = "SW_" + params['input_pipe']
                
            if (params['output_pipe'] in model_out_names):
                params['output_pipe'] = "SW_" + params['output_pipe']

        return self.template.format(**params)


class EmbeddingStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(Embedding)
        self.template = embed_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)
