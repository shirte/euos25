import random

from torch.utils.data import Sampler


class RoundRobinBatchSampler(Sampler):
    def __init__(
        self, samplers, offsets=None, start_index=0, randomize=False, rng=None
    ):
        self.samplers = samplers
        self.offsets = offsets or [0] * len(samplers)
        self.start_index = start_index
        self.randomize = randomize
        self.rng = rng or random
        self.num_batches = sum(len(s) for s in samplers)

    def __iter__(self):
        iters = [iter(s) for s in self.samplers]
        remaining = [len(s) for s in self.samplers]
        active = [i for i, r in enumerate(remaining) if r > 0]
        cursor = self.start_index % len(self.samplers)

        while active:
            if self.randomize:
                cursor = self.rng.choice(active)
            elif cursor not in active:
                cursor = active[0]

            batch = next(iters[cursor])
            remaining[cursor] -= 1
            offset = self.offsets[cursor]
            if offset:
                batch = [idx + offset for idx in batch]
            yield batch

            if remaining[cursor] == 0:
                active.remove(cursor)
                if not active:
                    break

            if not self.randomize:
                cursor = (cursor + 1) % len(self.samplers)

    def __len__(self):
        return self.num_batches
