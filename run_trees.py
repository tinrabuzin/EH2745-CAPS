import Trees

x = Trees.decision_tree.BasicTree()

p = x.calc_entropy(3)
print x.make_tree(x.data, x.classes, x.features)

print p