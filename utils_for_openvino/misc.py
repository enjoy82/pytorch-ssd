import time

#TODO
class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(key, "is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval