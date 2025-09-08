from abc import ABC, abstractmethod

class ModelingApproach(ABC):
    def __init__(self):
        self.data_preparer = None

    def set_data_preparer(self, data_preparer):
        self.data_preparer = data_preparer

    @abstractmethod
    def train(*args, **kwargs):
        pass

    @abstractmethod
    def predict(*args, **kwargs):
        pass

    def check_early_stopping(self, epoch, val_loss, val_roc_auc=None, val_balanced_acc=None, val_cohen_kappa=None, val_weighted_f1=None):
        if self.early_stopping_params['name'] == 'EarlyStopping':
            if self.early_stopping_params['monitor_metric'] == 'val_loss':
                early_stopping_metric = val_loss
                if 'min_or_max' not in self.early_stopping_params:
                    self.early_stopping.min_or_max = 'min'
            elif self.early_stopping_params['monitor_metric'] == 'val_roc_auc':
                early_stopping_metric = val_roc_auc
                if 'min_or_max' not in self.early_stopping_params:
                    self.early_stopping.min_or_max = 'max'
            elif self.early_stopping_params['monitor_metric'] == 'val_balanced_acc':
                early_stopping_metric = val_balanced_acc
                if 'min_or_max' not in self.early_stopping_params:
                    self.early_stopping.min_or_max = 'max'
            elif self.early_stopping_params['monitor_metric'] == 'val_cohen_kappa':
                early_stopping_metric = val_cohen_kappa
                if 'min_or_max' not in self.early_stopping_params:
                    self.early_stopping.min_or_max = 'max'
            elif self.early_stopping_params['monitor_metric'] == 'val_weighted_f1':
                early_stopping_metric = val_weighted_f1
                if 'min_or_max' not in self.early_stopping_params:
                    self.early_stopping.min_or_max = 'max'
            else:
                raise ValueError('Invalid monitor_metric in early_stopping_params')

            if 'min_or_max' in self.early_stopping_params:
                self.early_stopping.min_or_max = self.early_stopping_params['min_or_max']

            if self.early_stopping(
                current_metric=early_stopping_metric,
                epoch=epoch,
                model=self.model,
                tmp_dir=self.tmp_dir
                ):
                return True
        elif self.early_stopping_params['name'] == 'CombinedEarlyStopping':
            pass
        return False

