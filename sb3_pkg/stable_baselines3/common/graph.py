
class GenericGraphSpace:
    def __init__(self):
        pass

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return False

    def sample(self):
       raise Exception("sample from generic graph space not supported")

    def __eq__(self, other):
        return isinstance(other, GenericGraphSpace)

