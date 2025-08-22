import random
import torch
import logging
from tqdm import tqdm
from torch.utils.data import Sampler

# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

def split_by_task(dataset, task_col="task"):
    """Group dataset indices by task name."""
    task_groups = {}
    for i, ex in enumerate(dataset):
        task = ex[task_col].lower()
        task_groups.setdefault(task, []).append(i)
    return task_groups

class MultiTaskBatchSampler(Sampler):
    """Each batch has exactly one sample from each task."""
    def __init__(self, dataset, tasks, task_col="task"):
        self.dataset = dataset
        self.tasks = [t.lower() for t in tasks]
        self.task_groups = split_by_task(dataset, task_col)

        # Ensure all tasks exist
        for t in self.tasks:
            if t not in self.task_groups:
                raise ValueError(f"Task {t} not found in dataset.")

    def __iter__(self):
        # Shuffle indices inside each task group
        task_iters = {
            t: iter(random.sample(idxs, len(idxs)))
            for t, idxs in self.task_groups.items()
        }

        while True:
            batch = []
             # Shuffle the order of tasks for this batch
            tasks_shuffled = self.tasks.copy()
            random.shuffle(tasks_shuffled)
            for t in tasks_shuffled:
                try:
                    batch.append(next(task_iters[t]))
                except StopIteration:
                    return  # stop when one task runs out
            yield batch

    def __len__(self):
        return min(len(idxs) for idxs in self.task_groups.values())
    
def inspect_batches(dataloader, tokenizer, num_batches=5, decode=True):
    """
    Prints task composition and optionally decodes input_ids back to text
    for a few batches to verify sampler.
    """
    for i, batch in enumerate(dataloader):
        # Detect tasks
        if isinstance(batch, dict) and 'task' in batch:
            tasks_in_batch = batch['task']
        elif hasattr(batch, 'task'):
            tasks_in_batch = batch.task
        else:
            tasks_in_batch = batch.get('labels', None)  # fallback

        # Convert tensor to list
        if isinstance(tasks_in_batch, torch.Tensor):
            tasks_in_batch = tasks_in_batch.tolist()

        # Log tasks
        logger.info(f"Batch {i+1}: tasks = {tasks_in_batch}")

        # Decode input text if decode=True
        if decode and isinstance(batch, dict) and 'input_ids' in batch:
            input_ids = batch['input_ids']
            # If batched tensor, decode each example
            if isinstance(input_ids, torch.Tensor):
                for idx, ids in enumerate(input_ids):
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    logger.info(f"  Example {idx+1}: {text}")  # full text
            elif isinstance(input_ids, list):
                for idx, ids in enumerate(input_ids):
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    logger.info(f"  Example {idx+1}: {text}")  # full text

        if i + 1 >= num_batches:
            break