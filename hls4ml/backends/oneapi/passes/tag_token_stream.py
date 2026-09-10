from hls4ml.model.layers import Input
from hls4ml.model.optimizer import GlobalOptimizerPass


class TagTokenStream(GlobalOptimizerPass):
    def transform(self, model, node):
        if isinstance(node, Input):
            if not model.config.config['HLSConfig'].get('Autoregressive'):
                return  # Dont do anything for non-autoreg

            frontier = [node]
            visited = set()
            while frontier:
                node = frontier.pop()
                if node.name in visited:
                    continue
                visited.add(node.name)
                if not node.get_attr('token_stream', False):
                    node.set_attr('token_stream', True)

                output_map = node.get_output_use_map()
                for out_name, consumers in output_map.items():
                    in_shape = node.get_output_variable(out_name).shape
                    for consumer in consumers:
                        out_shape = consumer.get_output_variable().shape
                        if out_shape == in_shape:
                            frontier.append(consumer)
