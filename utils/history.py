class History:
    def __init__(self):
        self.data = {}

    def update(self, losses):
        for key, value in losses.items():
            self.data.setdefault(key, []).append(value)
    
    def get(self):
        return self.data
