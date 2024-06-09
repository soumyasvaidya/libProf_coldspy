class CCTNode:
    def __init__(self, frame=None):
        self.key = (frame.filename, frame.name, frame.lineno) if frame else ("root", "", 0)
        self.children = {}
        self.samples = 0

    def to_dict(self):
        return {
            "frame": self.key,
            "samples": self.samples,
            "children": {key: child.to_dict() for key, child in self.children.items()},
        }

class CCT:
    def __init__(self):
        self.root = CCTNode()

    def add_sample(self, stack):
        current = self.root
        for frame in stack:
            k = str((frame.filename, frame.name, frame.lineno))
            if k not in current.children:
                current.children[k] = CCTNode(frame)
            current = current.children[k]
        current.samples += 1
