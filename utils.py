import os

from torch.utils.tensorboard import SummaryWriter

class ModifiedTensorBoard:
    def __init__(self, log_dir='./tb_logs', start_step=0):
        subfolder_name = f'run_{len(os.listdir(log_dir))}'  # Unique run name
        run_dir = os.path.join(log_dir, subfolder_name)
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(run_dir)
        self.step = start_step

    def log_scalar(self, tag, value, step=None):
        if step is None:
            step = self.step
            self.step += 1
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, values_dict, step=None):
        if step is None:
            step = self.step
            self.step += 1
        for key, value in values_dict.items():
            self.writer.add_scalar(f"{tag}/{key}", value, step)

    def update_step(self, step):
        self.step = step

    def close(self):
        self.writer.close()
