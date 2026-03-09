class PoseGraph:

    def __init__(self):

        self.nodes = []

        self.edges = []

    def add_node(self,pose):

        self.nodes.append(pose)

        return len(self.nodes)-1


    def add_edge(self,i,j,T):

        self.edges.append((i,j,T))