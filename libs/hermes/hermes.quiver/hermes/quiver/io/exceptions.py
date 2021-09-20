class NoFilesFoundError(Exception):
    def __init__(self, path):
        super().__init__(f"Could not find path or paths matching {path}")
