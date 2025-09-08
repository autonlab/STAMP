import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, save_final_checkpoint=False):
        """
        Args:
            min_or_max (str): Whether to minimize or maximize the monitored metric.
            patience (int): Number of epochs to wait for a significant improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.min_or_max = None
        self.patience = patience
        self.min_delta = min_delta
        self.save_final_checkpoint = save_final_checkpoint
        self.best_metric = None
        self.best_epoch = 0
        self.counter = 0
        self.n_times_improved = 0

        self.random_seed = None

    def __call__(self, current_metric, epoch, model, tmp_dir, **kwargs):
        """
        Args:
            current_metric (float): The current value of the validation metric.
            epoch (int): The current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """

        improved = self.best_metric is None or (
            (self.min_or_max == 'min' and current_metric < self.best_metric - self.min_delta) or
            (self.min_or_max == 'max' and current_metric > self.best_metric + self.min_delta)
        )

        if improved:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.counter = 0  # Reset counter when there's an improvement
            self.n_times_improved += 1

            print(f'Saving best checkpoint at epoch {epoch}...')

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'current_metric': current_metric,
            }, tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')

            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.save_final_checkpoint:
                    # Save model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model.optimizer.state_dict(),
                        'current_metric': current_metric,
                    }, tmp_dir + f'/final_checkpoint_seed{self.random_seed}.pth')
                return True
            return False

class CombinedEarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, alpha=0.5, save_final_checkpoint=False):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            alpha (float): Weight for the loss and metric combination.
                           0.0 focuses only on AUC, 1.0 focuses only on loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.alpha = alpha
        self.save_final_checkpoint = save_final_checkpoint
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0

        self.shuffle_idx = None
        self.outer_fold_idx = None
        self.hp_set_idx = None
        self.inner_fold_idx = None

    def __call__(self, val_loss, val_auc, epoch, model, tmp_dir, **kwargs):
        """
        Args:
            val_loss (float): Validation loss for the current epoch.
            val_auc (float): Validation AUC for the current epoch.
            epoch (int): Current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        # Calculate the combined score
        combined_score = self.alpha * (-val_loss) + (1 - self.alpha) * val_auc

        if self.best_score is None or combined_score > self.best_score + self.min_delta:
            self.best_score = combined_score
            self.best_epoch = epoch
            self.counter = 0  # Reset patience counter

            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'loss': val_loss,
                    'auc': val_auc,
            }, tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')

            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.save_final_checkpoint:
                    # Save model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model.optimizer.state_dict(),
                        'loss': val_loss,
                        'auc': val_auc,
                    }, tmp_dir + f'/final_checkpoint_seed{self.random_seed}.pth')
                return True
            return False

class DualEarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, save_final_checkpoint=False):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_final_checkpoint = save_final_checkpoint
        self.best_loss = None
        self.best_auc = None
        self.best_loss_epoch = 0
        self.best_auc_epoch = 0

        self.shuffle_idx = None
        self.outer_fold_idx = None
        self.hp_set_idx = None
        self.inner_fold_idx = None

    def __call__(self, val_loss, val_auc, epoch, model, tmp_dir, **kwargs):
        """
        Args:
            val_loss (float): Validation loss for the current epoch.
            val_auc (float): Validation AUC for the current epoch.
            epoch (int): Current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        curr_improved = False
        # Check validation loss
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            curr_improved = True

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': val_loss,
                'auc': val_auc,
            }, tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')

        # Check validation AUC
        if self.best_auc is None or val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.best_auc_epoch = epoch

            # Prevent saving the same checkpoint in the current epoch
            if curr_improved == False:
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'loss': val_loss,
                    'auc': val_auc,
                }, tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')

        # Stop if neither has improved within the patience period
        if (epoch - self.best_loss_epoch >= self.patience and
            epoch - self.best_auc_epoch >= self.patience):
            if self.save_final_checkpoint:
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'loss': val_loss,
                    'auc': val_auc,
                }, tmp_dir + f'/final_checkpoint_seed{self.random_seed}.pth')
            return True
        return False

def build_early_stopping(params):
    name = params['name']
    if name == 'EarlyStopping':
        return EarlyStopping(patience=params['patience'], min_delta=params['min_delta'])
    elif name == 'CombinedEarlyStopping':
        return CombinedEarlyStopping(patience=params['patience'], min_delta=params['min_delta'], alpha=params['alpha'])
    elif name == 'DualEarlyStopping':
        return DualEarlyStopping(patience=params['patience'], min_delta=params['min_delta'])