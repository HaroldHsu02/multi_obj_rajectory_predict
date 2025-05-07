class T_tree:
    def __init__(self, name):
        # key是名字 child是子节点，support是支持度
        self.key = name
        self.child = []
        self.support = 1

    def get(self, getname):
        # 遍历每一个子节点
        for each_child in self.child:
            # 如果找到名字一样的
            if each_child.key == getname:
                # 那就准备返回这个节点
                return each_child
        return None

    def insert(self, newname):
        # 这里和get是一样的，只不过就是support+1
        if self.child:
            for each_child in self.child:
                if each_child.key == newname:
                    each_child.support += 1
                    return 0  # Indicates existing node support incremented
        # 如果没找到，那就新建一个对象，然后把对象放在子节点列表中
        node = T_tree(newname)
        self.child.append(node)
        return 1  # Indicates new node added


class New_Node_Support:
    def __init__(self):
        self.key = ""
        self.support = 1
