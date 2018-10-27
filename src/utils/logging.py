from tensorboardX import SummaryWriter

class TensorBoardWriter:

    def __init__(self, logdir):
        self.writer = SummaryWriter(logdir)

    def write_loss(self, labeled_losses, step):
        """
        param: labeled_losses - dictionary of loss values and their labels
            e.g. {"training": train_loss, "validation": valid_loss}
        """
        self.writer.add_scalars('loss', labeled_losses, step)

    def close(self):
        self.writer.close()

