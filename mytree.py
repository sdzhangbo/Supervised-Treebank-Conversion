class Tree(object):
    def __init__(self, index):
        self.parent = None
        self.index = index
        self.children = list()
        self.children_num = 0
        self._depth = -1
        self.order = list()

    def add_child(self, child):
        """
        :param child: a Tree object represent the child
        :return:
        """
        child.parent = self
        self.children.append(child)
        self.children_num += 1

    def size(self):
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.children_num):
            count += self.children[i].size()
        self._size = count
        return self._size


    def traverse(self):
        if len(self.order) > 0:
            return self.order

        for i in range(self.children_num):
            order = self.children[i].traverse()
            self.order.extend(order)
        self.order.append(self.index)
        return self.order

def creatTree(heads):
    tree = []
    #current sentence has already been numberized [form, head, rel]
    root = None
    for idx, head in enumerate(heads):
        tree.append(Tree(idx))

    for idx, head in enumerate(heads):
        if head == -1:
            root = tree[idx]
            continue
        assert head >= 0 and root is not None
        tree[head].add_child(tree[idx])
    return root, tree

if __name__ == '__main__':
    root, tree = creatTree([-1, 0, 1, 2, 2, 3])
    print(root.index)
    print(tree[root.index].traverse())
    print(root.size())


