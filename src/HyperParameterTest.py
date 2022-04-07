from keras_tuner import HyperParameters

hp = HyperParameters()
max_depth = 5
hp_depth = hp.Int("depth", 1, max_depth, default=4)
# hp.Int("layer0", 3, 4, parent_name="depth", parent_values=[1, 2, 3])
# foo = hp.Int("layer1", 2, hp.get("layer0"), parent_name="depth", parent_values=[2,3])
# if not (foo is None):
#     hp.Int("layer2", 1, foo, default=1, parent_name="depth", parent_values=[3])
for i in range(hp_depth):
    if i == 0:
        hp.Int("layer" + str(i), 3, 4)
    else:
        hp.Int("layer" + str(i), max(3 - i, 1), hp.get("layer" + str(i - 1)), parent_name="depth",
               parent_values=[*range(i + 1, max_depth+1)])
print(hp.values)
