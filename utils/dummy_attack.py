class DummyAttack():
    def __init__(self, m):
        ...

    def set_normalization_used(self, mean, std):
        ...

    def __call__(self, x, y):
        return x