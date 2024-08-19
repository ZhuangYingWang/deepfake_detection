import os


def create_dirs(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)
