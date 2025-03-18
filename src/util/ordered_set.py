from collections import OrderedDict
class OrderedSet(OrderedDict):
    def __init__(self, iterable=None):
        super().__init__()
        if iterable:
            for item in iterable:
                self[item] = None

    def add(self, item):
        self[item] = None

    def remove(self, item):
        try:
            del self[item]
        except KeyError:
            raise KeyError(f"Item {item} not found in OrderedSet") from None

    def discard(self, item):
        self.pop(item, None)

    def __contains__(self, item):
        return super().__contains__(item)

    def __len__(self):
        return super().__len__()

    def __iter__(self):
        return super().__iter__()

    def __repr__(self):
        elements = list(self.keys())
        return f"{self.__class__.__name__}({elements})"