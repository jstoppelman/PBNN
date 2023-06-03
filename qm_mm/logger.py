
class Logger:
    def __init__(self, fname):
        self.fname = fname

    def write(self, message):
        with open(self.fname, "a") as fh:
            fh.write(message)

