import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PreprocessedIterableDataset(IterableDataset):
    def __init__(
        self, data, tokenizer, batch_size, max_length, process_rank=0, world_size=1
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.process_rank = process_rank
        self.world_size = world_size

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Calculate the starting point and stride
        total_workers = num_workers * self.world_size
        worker_global_id = self.process_rank * num_workers + worker_id
        iter_data = itertools.islice(self.data, worker_global_id, None, total_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack(
            [item["attention_mask"].squeeze(0) for item in batch]
        )

        return {"input_ids": input_ids, "attention_mask": attention_mask}
