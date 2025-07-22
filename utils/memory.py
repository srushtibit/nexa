class ContextMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)
