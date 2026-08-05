"""The clone templates in the fpga backend are not enough for oneAPI, so this adds the missing parts"""

import shutil
from hls4ml.backends.fpga.passes.clone import Clone
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate

clone_stream_function_template = '{name}.async();'


class CloneTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Clone)

    def format(self, node):
        params = self._default_function_params(node)
        for i in range(len(node.outputs)):
            params[f'output{i + 1}_pipe'] = node.variables[node.outputs[i]].pipe_name

        output_pipes = ', '.join([f'{{output{i + 1}_pipe}}' for i in range(len(node.outputs))])

        template = f'task_sequence<nnet::clone_stream<{{input_pipe}}, {output_pipes}, {{size}}>> {{name}};'

        max_invoc = node.model.config.get_config_value('HLSConfig').setdefault('MaxInvoc', None)
        if max_invoc is not None:
            
            template = (
                f'task_sequence<nnet::clone_stream<{{input_pipe}}, {output_pipes}, {{size}}>, {{maxinvoc}}>  {{name}};'
            )

            params['maxinvoc'] = 'ts_invoc_props' if shutil.which('ahls') else f'{max_invoc},{max_invoc}'
    
        autoreg_model: bool = node.model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None

        if autoreg_model:
            model_inp_names = [layer.pipe_name for layer in node.model.get_input_variables()] 
            model_out_names = [layer.pipe_name for layer in node.model.get_output_variables()]

            out_interface = False
            for i in range(len(node.outputs)):
                if params[f'output{i + 1}_pipe'] in model_out_names:
                    out_interface = True 
                    break

            if (params['input_pipe'] in model_inp_names):
                params['input_pipe'] = "SW_" + params['input_pipe']
            
            elif out_interface:
                for i in range(len(node.outputs)):
                    params[f'output{i + 1}_pipe'] = "SW_" + params[f'output{i + 1}_pipe']

        return template.format(**params)


class CloneStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(Clone)
        self.template = clone_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)
