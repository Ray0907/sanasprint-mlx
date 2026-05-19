def test_package_imports():
    import sanasprint_mlx

    assert sanasprint_mlx.__version__


def test_version_is_non_empty_string():
    from sanasprint_mlx import __version__

    assert isinstance(__version__, str)
    assert __version__
