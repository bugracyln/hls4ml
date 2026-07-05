"""
These are the stream oneAPI templates for embedding layers. The io_parallel ones are in backends/fpga/passes/embedding.py.
"""

from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Embedding

embed_config_template = """struct config{index} : nnet::embed_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned vocab_size = {vocab_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const unsigned num_banks = DIV_ROUNDUP(n_out, reuse_factor);
    typedef {embeddings_t.name} embeddings_t;
    [[intel::fpga_memory, intel::numbanks(num_banks),
    intel::bankwidth(sizeof(embeddings_t::value_type))]] static constexpr embeddings_t embeddings = {e};
}};\n"""


embed_function_template = 'nnet::embedding<{input_t}, {output_t}, {config}>({input}, {output});'

embed_include_list = ['nnet_utils/nnet_embed.h', 'nnet_utils/nnet_embed_stream.h']

embed_task_sequence_template = 'task_sequence<nnet::embedding_stream<{input_pipe}, {output_pipe}, {config}>> {name};'

embed_task_sequence_template_max_invoc = (
    'task_sequence<nnet::embedding_stream<{input_pipe}, {output_pipe}, {config}>,ts_invoc_props> {name};'
)

embed_stream_function_template = '{name}.async();'


class EmbeddingConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Embedding)
        self.template = embed_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['e'] = node.get_weights('embeddings').name

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
            params['maxInvoc'] = max_invoc
        
        autoreg_model: bool = node.model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None

        if autoreg_model:
            model_inp_names = [layer.pipe_name for layer in node.model.get_input_variables()] 
            model_out_names = [layer.pipe_name for layer in node.model.get_output_variables()]

            if (params['input_pipe'] in model_inp_names):
                params['input_pipe'] = "SW_" + params['input_pipe']
                
            elif (params['output_pipe'] in model_out_names):
                params['output_pipe'] = "SW_" + params['output_pipe']

        return self.template.format(**params)


class EmbeddingStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(Embedding)
        self.template = embed_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)
