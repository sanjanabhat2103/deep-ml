def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	total_params = 0
    for layer in layers:
        subtotal = 0
        bias = 1 if layer.get('bias', True) else 0
        if layer['type'] == 'conv2d':
            k = layer['kernel_size']
            subtotal += (layer['in_channels'] * k * k + bias) * layer['out_channels']
        elif layer['type'] == 'dense':
            subtotal += (layer['input_size'] + bias) * layer['output_size']
        total_params += subtotal
    return total_params