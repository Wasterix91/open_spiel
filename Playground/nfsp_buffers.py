import random
from collections import deque

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.count = 0

    def add(self, item):
        self.count += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            index = random.randint(0, self.count - 1)
            if index < self.capacity:
                self.data[index] = item

    def sample(self, batch_size):
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self):
        return len(self.data)
