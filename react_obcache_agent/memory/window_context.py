class WindowContextManager:
    def __init__(self, keep_last_steps: int = 3):
        self.keep_last_steps = keep_last_steps

    def select(self, trajectory):
        keep_items = self.keep_last_steps * 2
        return trajectory[-keep_items:]
