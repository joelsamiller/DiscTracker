from collections import OrderedDict

objects = OrderedDict()

vals = [(1,2)]

objects[0] = vals
vals = [(2,2)]
objects[1] = vals

objects[0].append((3,4))
objects[1].append((4,5))

vals = [vals[-1] for vals in list(objects.values())]

print(len(objects.keys()))
print(objects[0][-1][0])

print(list(objects.keys())[-1])
print("done")