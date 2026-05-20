from sanasprint_mlx.memory.mlx_cache import mlx_cache_limit, trim_mlx_cache


class FakeMLX:
    def __init__(self):
        self.events = []

    def set_cache_limit(self, value):
        self.events.append(("set_cache_limit", value))
        return 123

    def clear_cache(self):
        self.events.append(("clear_cache",))


def test_trim_mlx_cache_sets_zero_limit_and_clears_cache():
    fake = FakeMLX()

    previous = trim_mlx_cache(fake)

    assert previous == 123
    assert fake.events == [("set_cache_limit", 0), ("clear_cache",)]


def test_mlx_cache_limit_restores_previous_limit_after_clear():
    fake = FakeMLX()

    with mlx_cache_limit(0, fake):
        fake.events.append(("body",))

    assert fake.events == [
        ("set_cache_limit", 0),
        ("body",),
        ("clear_cache",),
        ("set_cache_limit", 123),
    ]
