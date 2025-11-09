import pandas as pd
import torch
import torch.nn as nn
from os import (remove as os_remove, path as os_path)
from tqdm import tqdm
from stamp.modeling.modeling_approach import ModelingApproach
from stamp.modeling.early_stopping import *
from stamp.modeling.utils import calculate_binary_performance_metrics, calculate_multiclass_performance_metrics, get_cbramod_model

class CBraModModelingApproach(ModelingApproach):
    def __init__(
        self,
        dataset_name,
        n_epochs,
        train_batch_size,
        test_batch_size,
        min_epoch,
        lr_params,
        optimizer_params,
        problem_type,
        n_classes,
        early_stopping_params,
        model_params,
        device,
        debug_size=None,
        use_tqdm=True,
        store_attention_weights=False,
        use_gradient_clipping=False,
        label_smoothing=None,
        **kwargs
        ):

        super().__init__()

        self.random_seed=None
        self.dataset_name = dataset_name
        self.n_epochs = n_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.min_epoch = min_epoch
        self.lr_params = lr_params
        self.optimizer_params = optimizer_params
        self.debug_size = debug_size
        self.use_tqdm = use_tqdm
        self.store_attention_weights = store_attention_weights
        self.use_gradient_clipping = use_gradient_clipping
        self.problem_type = problem_type
        self.n_classes = n_classes
        self.tmp_dir = early_stopping_params['tmp_dir']
        self.label_smoothing = label_smoothing

        self.model_params = model_params
        self.model_params['num_of_classes'] = n_classes
        self.model_params['cuda'] = device

        print(self.model_params)

        self.model = get_cbramod_model(model_params, dataset_name)
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model.to(self.device)

        self.backbone_params = []
        self.other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                self.backbone_params.append(param)

                if self.model_params['frozen']:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                self.other_params.append(param)

    def train(self, train_data_loader, val_data_loader):

        if self.problem_type == 'binary':
            criterion = nn.BCEWithLogitsLoss()
        elif self.problem_type == 'multiclass':
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise ValueError()
        if self.model_params['multi_lr']:
            self.model.optimizer = torch.optim.AdamW([
                        {'params': self.backbone_params, 'lr': self.lr_params['initial_lr']},
                        {'params': self.other_params, 'lr': self.lr_params['initial_lr'] * 5}
                    ], weight_decay=self.optimizer_params['weight_decay'])
        else:
            self.model.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                        lr=self.lr_params['initial_lr'], 
                                        weight_decay=self.optimizer_params['weight_decay']
                                        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.model.optimizer,
                    T_max=self.n_epochs * len(train_data_loader),
                    eta_min=1e-6
                )

        # Initialize lists to store metrics
        self.initialize_train_val_performance_lists()

        self.val_cohen_kappa_best = 0
        self.val_roc_auc_best = 0
        self.best_epoch = 0
        for epoch in range(self.n_epochs):  # Number of epochs
            print(f'Epoch: {epoch}')
            # Training
            self.model.train()
            epoch_train_main_loss = 0.0
            train_probs = []
            train_preds = []
            train_labels = []
            for seq_batch, label_batch, _ in tqdm(train_data_loader, mininterval=10):
                label_batch = self.correct_tuev_label_batch(label_batch) # IMPORTANT for TUEV dataset

                probs, preds, main_loss = self.evaluate_batch(
                    seq_batch=seq_batch,
                    label_batch=label_batch,
                    mode='train',
                    criterion=criterion
                )

                # print(main_loss)
                main_loss.backward()
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.model.optimizer.step()

                if self.lr_params['use_scheduler']:
                    self.scheduler.step()

                train_probs.append(probs.detach().cpu())
                train_preds.append(preds.detach().cpu())
                train_labels.append(label_batch.detach().cpu())

                epoch_train_main_loss += main_loss.item()

            train_main_loss = epoch_train_main_loss / len(train_data_loader)

            if self.device.type == 'cuda':
                seq_batch = seq_batch.cpu()
                label_batch = label_batch.cpu()
                probs = probs.cpu() if self.problem_type == 'binary' else None
                main_loss = main_loss.cpu()
                del seq_batch, label_batch, probs
                torch.cuda.empty_cache()

            self.evaluate_split(split_name='train', truths=torch.cat(train_labels), preds=torch.cat(train_preds), probs=torch.cat(train_probs) if self.problem_type == 'binary' else None)
            self.train_main_losses.append(train_main_loss)

            # Validation
            self.model.eval()
            epoch_val_main_loss = 0.0
            val_probs = []
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for seq_batch, label_batch, _ in tqdm(val_data_loader, mininterval=1):
                    label_batch = self.correct_tuev_label_batch(label_batch) # IMPORTANT for TUEV dataset

                    probs, preds, main_loss = self.evaluate_batch(
                        seq_batch=seq_batch,
                        label_batch=label_batch,
                        mode='val',
                        criterion=criterion
                    )

                    val_probs.append(probs.cpu())
                    val_preds.append(preds.cpu())
                    val_labels.append(label_batch.cpu())
                    epoch_val_main_loss += main_loss.item()

            val_main_loss = epoch_val_main_loss / len(val_data_loader)

            self.evaluate_split(split_name='val', truths=torch.cat(val_labels), preds=torch.cat(val_preds), probs=torch.cat(val_probs))
            self.val_main_losses.append(val_main_loss)

            print(f'train_loss: {train_main_loss:.4f}, val_loss: {val_main_loss:.4f}')
            print(f'train_balanced_acc: {self.train_balanced_acc:.4f}, val_balanced_acc: {self.val_balanced_acc:.4f}')
            if self.problem_type == 'binary':
                print(f'train_pr_auc: {self.train_pr_auc:.4f}, val_pr_auc: {self.val_pr_auc:.4f}')
                print(f'train_roc_auc: {self.train_roc_auc:.4f}, val_roc_auc: {self.val_roc_auc:.4f}')

                if self.val_roc_auc > self.val_roc_auc_best:
                    self.best_epoch = epoch
                    self.val_roc_auc_best = self.val_roc_auc
                    print(f'roc auc improved on epoch {epoch}, saving...')
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model.optimizer.state_dict(),
                    'current_metric': self.val_roc_auc,
                }, self.tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')
            elif self.problem_type == 'multiclass':
                print(f'train_cohen_kappa: {self.train_cohen_kappa:.4f}, val_cohen_kappa: {self.val_cohen_kappa:.4f}')
                print(f'train_weighted_f1: {self.train_weighted_f1:.4f}, val_weighted_f1: {self.val_weighted_f1:.4f}')

                if self.val_cohen_kappa > self.val_cohen_kappa_best:
                    self.best_epoch = epoch
                    self.val_cohen_kappa_best = self.val_cohen_kappa
                    print(f'kappa score improved on epoch {epoch}, saving...')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.model.optimizer.state_dict(),
                        'current_metric': self.val_cohen_kappa,
                    }, self.tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')

            if self.device.type == 'cuda':
                seq_batch = seq_batch.cpu()
                label_batch = label_batch.cpu()
                probs = probs.cpu()
                main_loss = main_loss.cpu()
                del seq_batch, label_batch, probs, train_probs, train_labels, val_probs, val_labels
                torch.cuda.empty_cache()

        # Load the best model
        assert os_path.exists(self.tmp_dir), 'Tmp dir does not exist.'
        print(f'Loading best checkpoint from epoch {self.best_epoch}...')
        checkpoint = torch.load(self.tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint

        # Remove the early stopping checkpoint
        os_remove(self.tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')

        if self.device.type == 'cuda':
            del train_data_loader, val_data_loader
            torch.cuda.empty_cache()

    def predict(
        self,
        test_data_loader
        ):

        self.model.eval()
        test_probs = []
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for seq_batch, label_batch, _ in test_data_loader:
                label_batch = self.correct_tuev_label_batch(label_batch) # IMPORTANT for TUEV dataset
                
                probs, preds, _ = self.evaluate_batch(
                    seq_batch=seq_batch,
                    label_batch=None,
                    mode='test',
                    criterion=None
                )
                test_probs.append(probs.cpu())
                test_preds.append(preds.cpu())
                test_labels.extend(label_batch.numpy())

        if self.problem_type == 'binary':
            prob_df = pd.DataFrame({'prob': torch.cat(test_probs).cpu().numpy()})
            prob_df.columns = ['prob']
        elif self.problem_type == 'multiclass':
            prob_df= pd.DataFrame(torch.cat(test_probs).cpu().numpy())
            prob_df.columns = [f'prob_class_{i}' for i in range(prob_df.shape[1])]
        else:
            prob_df = None

        pred_df = pd.DataFrame({'pred': torch.cat(test_preds).cpu().numpy()})
        pred_df.columns = ['pred']

        extra_info = {
            'best_epoch': self.best_epoch,
            'train_main_losses': self.train_main_losses,
            'train_balanced_acc_list': self.train_balanced_acc_list,
            'train_roc_auc_list': self.train_roc_auc_list,
            'train_pr_auc_list': self.train_pr_auc_list,
            'train_cohen_kappa_list': self.train_cohen_kappa_list,
            'train_weighted_f1_list': self.train_weighted_f1_list,
            'train_cm_list': self.train_cm_list,
            'val_main_losses': self.val_main_losses,
            'val_balanced_acc_list': self.val_balanced_acc_list,
            'val_roc_auc_list': self.val_roc_auc_list,
            'val_pr_auc_list': self.val_pr_auc_list,
            'val_cohen_kappa_list': self.val_cohen_kappa_list,
            'val_weighted_f1_list': self.val_weighted_f1_list,
            'val_cm_list': self.val_cm_list,
            'prob_df': prob_df,
            'test_labels': test_labels
        }

        self.model.cpu()
        torch.cuda.empty_cache()

        return pred_df, extra_info

    def correct_tuev_label_batch(self, label_batch):
        if self.dataset_name == 'tuev':
            # The TUEV labels are 1-indexed, so we need to shift them to be 0-indexed
            label_batch = label_batch - 1 # Shift labels to be 0-indexed
        return label_batch

    def initialize_train_val_performance_lists(self):
        self.train_main_losses = []
        self.train_balanced_acc_list = []
        self.train_pr_auc_list = []
        self.train_roc_auc_list = []
        self.train_cohen_kappa_list = []
        self.train_weighted_f1_list = []
        self.train_cm_list = []

        self.val_main_losses = []
        self.val_pr_auc_list = []
        self.val_roc_auc_list = []
        self.val_balanced_acc_list = []
        self.val_cohen_kappa_list = []
        self.val_weighted_f1_list = []
        self.val_cm_list = []

    def evaluate_batch(
        self,
        seq_batch,
        label_batch,
        mode,
        criterion
        ):
        
        # Move tensors to the specified device
        seq_batch = seq_batch.to(self.device) # Shape: (batch_size, max_hr, n_channels, n_features)

        if mode == 'train':
            self.model.optimizer.zero_grad()

        logits = self.model(x=seq_batch) # Binary shape: (batch_size, 1), Multiclass shape: (batch_size, n_classes)
        if self.problem_type == 'binary':
            logits = logits.squeeze()  # Remove class dimension for binary
            if label_batch is not None:
                label_batch = label_batch.float()

        # Make sure each tensor has atleast 1 dim to prevent error
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)

        if criterion is not None:
            label_batch = label_batch.to(self.device) # Shape: (batch_size)
            loss = criterion(logits, label_batch) # Single value
        else:
            loss = None

        # Run outputs through sigmoid to get probabilities
        if self.problem_type == 'binary':
            probs = torch.sigmoid(logits)
            preds = torch.gt(probs, 0.5).long()
        elif self.problem_type == 'multiclass':
            probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
            preds = torch.argmax(logits, dim=-1)   # Get predicted class indices

        return probs, preds, loss

    def evaluate_split(self, split_name, truths, preds, probs=None):
        """
        Evaluate metrics for a given split (train/val/test) and update lists.
        Args:
            split_name (str): 'train', 'val', or 'test'
            truths (array-like): Ground-truth labels
            preds (array-like): Predicted labels
            probs (array-like or None): Predicted probabilities (binary only)
        """
        if self.problem_type == 'binary':
            balanced_acc, pr_auc, roc_auc, cm = calculate_binary_performance_metrics(
                truths=truths,
                probs=probs,
                preds=preds
            )

            # Dynamically choose which lists to update
            getattr(self, f"{split_name}_pr_auc_list").append(pr_auc)
            getattr(self, f"{split_name}_roc_auc_list").append(roc_auc)

            setattr(self, f"{split_name}_pr_auc", pr_auc)
            setattr(self, f"{split_name}_roc_auc", roc_auc)
            setattr(self, f"{split_name}_cohen_kappa", None)
            setattr(self, f"{split_name}_weighted_f1", None)

        elif self.problem_type == 'multiclass':
            balanced_acc, cohen_kappa, weighted_f1, cm = calculate_multiclass_performance_metrics(
                truths=truths,
                preds=preds
            )

            getattr(self, f"{split_name}_cohen_kappa_list").append(cohen_kappa)
            getattr(self, f"{split_name}_weighted_f1_list").append(weighted_f1)

            setattr(self, f"{split_name}_pr_auc", None)
            setattr(self, f"{split_name}_roc_auc", None)
            setattr(self, f"{split_name}_cohen_kappa", cohen_kappa)
            setattr(self, f"{split_name}_weighted_f1", weighted_f1)

        else:
            raise ValueError(f"Invalid problem_type: {self.problem_type}")

        getattr(self, f"{split_name}_balanced_acc_list").append(balanced_acc)
        getattr(self, f"{split_name}_cm_list").append(cm)

        setattr(self, f"{split_name}_balanced_acc", balanced_acc)
        setattr(self, f"{split_name}_cm", cm)