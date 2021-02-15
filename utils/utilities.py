class ObjFromDict:
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [ObjFromDict(x) if isinstance(
                    x, dict) else x for x in b])
            else:
                setattr(self, a, ObjFromDict(b) if isinstance(b, dict) else b)
