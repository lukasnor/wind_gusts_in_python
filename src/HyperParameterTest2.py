from keras_tuner import HyperParameters

hp_input_size = 10
hp_degree = 12
min_layer_size = max(hp_input_size / 4, hp_degree)
max_layer_size = hp_input_size * 4
layer_step = int(hp_input_size / 2)
possible_layer_sizes = [*range(min_layer_size, max_layer_size + 1, layer_step)]

possible_layer_size_ranges = ([possible_layer_sizes] + [possible_layer_sizes[:-i] for i in
                                                       range(1, len(possible_layer_sizes))]).reverse()
hp = HyperParameters()
layer0 = hp.Choice('layer0_size', values=possible_layer_sizes)
min_depth = 2
max_depth = 5
depth = hp.Int("depth", min_depth, max_depth)
if 1 < depth:
    iter_count = 0
    for layer_index in range(depth - 1, -1, -1):
        if iter_count == 0:
            parent_units_name = "layer0_size"
            parent_units_value = possible_layer_sizes[layer_index]
            child_units_name = 'units_layer_' + str(depth) + str(layer_index)
            child_units_value = parent_units_value
        else:
            parent_units_name = 'units_layer_' + str(depth) + str(layer_index + 1)
            parent_units_value = possible_layer_sizes[layer_index + 1]
            child_units_name = 'units_layer_' + str(depth) + str(layer_index)
            child_units_value = possible_layer_sizes[layer_index]

        # Add and Activate child HP under parent HP using conditional scope
        with hp.conditional_scope(parent_units_name, parent_units_value):
            hidden_units = hp.Choice(child_units_name, values=child_units_value)

