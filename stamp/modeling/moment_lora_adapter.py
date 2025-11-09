import pandas as pd
import torch
import torch.nn as nn
from os import (remove as os_remove, path as os_path)
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from momentfm import MOMENTPipeline

from stamp.modeling.modeling_approach import ModelingApproach
from stamp.modeling.early_stopping import *
from stamp.modeling.utils import calculate_binary_performance_metrics, calculate_multiclass_performance_metrics
from stamp.modeling.stamp import STAMP

class FullModel(nn.Module):
    def __init__(self, moment_model, adapter):
        super().__init__()
        self.moment_model = moment_model
        self.tsfm_adapter = adapter
        self.n_temporal_channels = adapter.n_temporal_channels
        self.n_spatial_channels = adapter.n_spatial_channels
        self.channel_product = self.n_temporal_channels * self.n_spatial_channels
    
    def forward(self, x, mask, return_attention, chunk_size):
        # x shape: (batch_size * n_temporal_channels * n_spatial_channels, 1, 512)
        # mask_shape: (batch_size * n_temporal_channels * n_spatial_channels, 512)

        batch_size = x.shape[0] // (self.n_temporal_channels * self.n_spatial_channels) # Calculate batch size

        if chunk_size is None:
            chunk_size = x.shape[0]  # Process all at once (current behavior)

        batch_size = x.shape[0] // (self.n_temporal_channels * self.n_spatial_channels)
        all_embeddings = []
        
        # Process in chunks with explicit memory management
        for i in range(0, x.shape[0], chunk_size):
            # Create chunk views (no memory copy)
            chunk_x = x[i:i+chunk_size, :, :].contiguous()
            chunk_mask = mask[i:i+chunk_size, :].contiguous()
            
            # Process chunk with gradient checkpointing if available
            with torch.amp.autocast('cuda', enabled=True):
                if hasattr(self.moment_model, 'gradient_checkpointing') and self.moment_model.gradient_checkpointing:
                    chunk_embeddings = torch.utils.checkpoint.checkpoint(
                        self.moment_model, chunk_x, chunk_mask, use_reentrant=False
                    ).embeddings
                else:
                    chunk_embeddings = self.moment_model(x_enc=chunk_x, input_mask=chunk_mask).embeddings
            
            # Store embeddings and clean up immediately
            all_embeddings.append(chunk_embeddings.clone())  # Clone to detach from computation graph
            
            # Explicit cleanup
            del chunk_x, chunk_mask, chunk_embeddings
            
            # Force garbage collection every few chunks
            if (i // chunk_size) % 4 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0) # Shape: (batch_size * n_temporal_channels * n_spatial_channels, embedding_dim)
        embeddings = embeddings.view(batch_size, -1) # Shape: (batch_size, n_temporal_channels * n_spatial_channels * embedding_dim)

        embedding_dim = embeddings.shape[1] // self.channel_product
        
        # Reshape to (batch_size, n_spatial * n_temporal, embedding_dim)
        embeddings = embeddings.reshape(embeddings.shape[0], self.channel_product, embedding_dim) # Shape: (batch_size, n_temporal_channels * n_spatial_channels, embedding_dim)
        
        # Then reshape to (batch_size, n_spatial, n_temporal, embedding_dim)
        embeddings = embeddings.reshape(embeddings.shape[0], self.n_spatial_channels, self.n_temporal_channels, embedding_dim) # Shape: (batch_size, n_spatial_channels, n_temporal_channels, embedding_dim)

        embeddings = embeddings.permute(0, 2, 1, 3)  # Shape: (batch_size, n_temporal_channels, n_spatial_channels, embedding_dim)
        
        output = self.tsfm_adapter(embeddings, return_attention=return_attention)
        return output

class MOMENTLoraAdapterModelingApproach(ModelingApproach):
    def __init__(
        self,
        input_dim,
        D,
        n_temporal_channels,
        n_spatial_channels,
        encoder_aggregation,
        use_batch_norm,
        use_instance_norm,
        initial_proj_params,
        final_classifier_params,
        pe_params,
        transformer_params,
        gated_mlp_params,
        mhap_params,
        n_epochs,
        train_batch_size,
        test_batch_size,
        min_epoch,
        early_stopping_params,
        checkpointing_params,
        lr_params,
        optimizer_params,
        problem_type,
        n_classes,
        device,
        moment_model_name,
        moment_models_dir,
        lora_params,
        chunk_size,
        n_cls_tokens=None,
        debug_size=None,
        use_tqdm=True,
        store_attention_weights=False,
        use_gradient_clipping=False,
        label_smoothing=None,
        **kwargs
        ):

        super().__init__()

        self.random_seed=None
        self.input_dim = input_dim
        self.D = D
        self.n_temporal_channels = n_temporal_channels
        self.n_spatial_channels = n_spatial_channels
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        self.n_cls_tokens = n_cls_tokens
        self.encoder_aggregation = encoder_aggregation
        self.initial_proj_params = initial_proj_params
        self.final_classifier_params = final_classifier_params
        self.transformer_params = transformer_params
        self.gated_mlp_params = gated_mlp_params
        self.mhap_params = mhap_params
        self.pe_params = pe_params
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
        self.label_smoothing = label_smoothing
        self.chunk_size = chunk_size

        if early_stopping_params is not None:
            self.use_early_stopping = True
            self.early_stopping = build_early_stopping(early_stopping_params)
            self.early_stopping_params = early_stopping_params
            if self.problem_type == 'binary':
                self.early_stopping_params['monitor_metric'] = 'val_roc_auc'
            elif self.problem_type == 'multiclass':
                self.early_stopping_params['monitor_metric'] = 'val_cohen_kappa'
            else:
                raise ValueError()
            self.tmp_dir = early_stopping_params.get('tmp_dir')
        else:
            self.use_early_stopping = False
            self.early_stopping = None
            self.tmp_dir = None

        self.checkpointing_params = checkpointing_params

        moment_model = MOMENTPipeline.from_pretrained(
            f"AutonLab/{moment_model_name}",
            model_kwargs={
                "task_name": "embedding",
                'freeze_encoder': False,
                'freeze_embedder': False,
                'enable_gradient_checkpointing': True
            },
            # Give local dir
            # to avoid downloading the model every time
            cache_dir = moment_models_dir,
            local_files_only=True
        )

        moment_model.init(); # NOTE: IMPORTANT!!! Otherwise, the model will not be initialized properly.
        
        config_dict = vars(moment_model.config)
        moment_model.config = type('Config', (), config_dict)()
        moment_model.config.get = lambda key, default=None: getattr(moment_model.config, key, default)

        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=lora_params['alpha'],
            target_modules=['q','v'],
            lora_dropout=lora_params['dropout'],
            )

        moment_model = get_peft_model(moment_model, lora_config)
        moment_model.print_trainable_parameters() # Be sure only LoRA params are trainable
        
        adapter = STAMP(
            use_batch_norm=self.use_batch_norm,
            use_instance_norm=self.use_instance_norm,
            input_dim=self.input_dim,
            D=self.D,
            n_temporal_channels=self.n_temporal_channels,
            n_spatial_channels=self.n_spatial_channels,
            initial_proj_params=self.initial_proj_params,
            pe_params=self.pe_params,
            transformer_params=self.transformer_params,
            gated_mlp_params=self.gated_mlp_params,
            encoder_aggregation=self.encoder_aggregation,
            mhap_params=self.mhap_params,
            final_classifier_params=self.final_classifier_params,
            n_classes=self.n_classes,
            n_cls_tokens=self.n_cls_tokens,
        )

        self.model = FullModel(moment_model, adapter)

        self.device = torch.device(device)
        self.model.to(self.device)

    def train(
        self,
        train_data_loader,
        val_data_loader):

        # Set random seed for reproducibility
        torch.manual_seed(self.random_seed)

        if self.use_early_stopping:
            self.early_stopping.random_seed = self.random_seed

        # Initialize the criterion
        if self.problem_type == 'binary':
            criterion = nn.BCEWithLogitsLoss()
        elif self.problem_type == 'multiclass':
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        elif self.problem_type == 'regression':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown main loss type: {self.main_loss_type}")

        self.initialize_optimizer()

        # Setup learning rate scheduler
        if self.lr_params['use_scheduler']:
            if self.lr_params['scheduler_type'] == 'one_cycle':
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.model.optimizer,
                    max_lr=self.lr_params['max_lr'],
                    total_steps=self.n_epochs * len(train_data_loader)
                )
            elif self.lr_params['scheduler_type'] == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.model.optimizer,
                    T_max=self.n_epochs * len(train_data_loader),
                    eta_min=self.lr_params['eta_min']
                )
            else:
                raise NotImplementedError(f"Scheduler type {self.lr_params['scheduler_type']} not implemented.")

        # Initialize lists to store metrics
        self.initialize_train_val_performance_lists()
        train_iterator, val_iterator = self.initialize_train_val_iterators(train_data_loader, val_data_loader)

        stopped_early = False
        for epoch in range(self.n_epochs):  # Number of epochs
            print(f'Epoch: {epoch}')
            # Training
            self.model.train()
            epoch_train_main_loss = 0.0
            train_probs = []
            train_preds = []
            train_labels = []
            print('Training...')
            for seq_batch, label_batch, mask_batch, sample_key_batch in train_iterator:

                if self.problem_type == 'binary':
                    label_batch = label_batch.to(torch.float32)
                elif self.problem_type == 'multiclass':
                    label_batch = label_batch.to(torch.long)
                else:
                    pass

                probs, preds, main_loss, _ = self.evaluate_batch(
                    seq_batch=seq_batch,
                    label_batch=label_batch,
                    mask_batch=mask_batch,
                    mode='train',
                    criterion=criterion,
                    chunk_size=self.chunk_size
                )

                main_loss.backward()
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.model.optimizer.step()

                if self.lr_params['use_scheduler'] and self.lr_params['scheduler_type'] == 'one_cycle':
                    self.scheduler.step()

                if self.problem_type == 'binary':
                    train_probs.append(probs.detach().cpu())
                train_preds.append(preds.detach().cpu())
                train_labels.append(label_batch.detach().cpu())

                # NOTE: We only call .item() after the backward pass because it removes the gradients
                epoch_train_main_loss += main_loss.item()

            if self.lr_params['use_scheduler'] and self.lr_params['scheduler_type'] != 'one_cycle':
                self.scheduler.step()

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
                for seq_batch, label_batch, mask_batch, sample_key_batch in val_iterator:

                    if self.problem_type == 'binary':
                        label_batch = label_batch.to(torch.float32)
                    elif self.problem_type == 'multiclass':
                        label_batch = label_batch.to(torch.long)
                    else:
                        pass

                    probs, preds, main_loss, _ = self.evaluate_batch(
                        seq_batch=seq_batch,
                        label_batch=label_batch,
                        mask_batch=mask_batch,
                        mode='val',
                        criterion=criterion,
                        chunk_size=self.chunk_size
                    )

                    val_probs.append(probs.cpu())
                    val_preds.append(preds.cpu())
                    val_labels.append(label_batch.cpu())
                    epoch_val_main_loss += main_loss.item()

            val_main_loss = epoch_val_main_loss / len(val_data_loader)

            self.evaluate_split(split_name='val', truths=torch.cat(val_labels), preds=torch.cat(val_preds), probs=torch.cat(val_probs))
            self.val_main_losses.append(val_main_loss)

            print(f'train_main_loss: {train_main_loss:.4f}, val_loss: {val_main_loss:.4f}')
            print(f'train_balanced_acc: {self.train_balanced_acc:.4f}, val_balanced_acc: {self.val_balanced_acc:.4f}')
            if self.problem_type == 'binary':
                print(f'train_pr_auc: {self.train_pr_auc:.4f}, val_pr_auc: {self.val_pr_auc:.4f}')
                print(f'train_roc_auc: {self.train_roc_auc:.4f}, val_roc_auc: {self.val_roc_auc:.4f}')
            elif self.problem_type == 'multiclass':
                print(f'train_cohen_kappa: {self.train_cohen_kappa:.4f}, val_cohen_kappa: {self.val_cohen_kappa:.4f}')
                print(f'train_weighted_f1: {self.train_weighted_f1:.4f}, val_weighted_f1: {self.val_weighted_f1:.4f}')

            print(f'Val CM:\n{self.val_cm}')

            if self.use_early_stopping and epoch > self.min_epoch:
                stopped_early = self.check_early_stopping(epoch, val_loss=val_main_loss, val_roc_auc=self.val_roc_auc, val_balanced_acc=self.val_balanced_acc,
                                                          val_cohen_kappa=self.val_cohen_kappa, val_weighted_f1=self.val_weighted_f1)
                if stopped_early:
                    break

            if self.device.type == 'cuda':
                seq_batch = seq_batch.cpu()
                label_batch = label_batch.cpu()
                probs = probs.cpu()
                main_loss = main_loss.cpu()
                del seq_batch, label_batch, probs, train_probs, train_labels, val_probs, val_labels
                torch.cuda.empty_cache()

        if self.checkpointing_params is not None:
            # make checkpoint dir
            if self.checkpointing_params.get('checkpoint_dir') is not None:
                self.save_checkpoint(
                    epoch=epoch,
                    train_loss=train_main_loss,
                    val_loss=val_main_loss,
                    checkpoint_path=self.checkpointing_params.get('checkpoint_dir') + f'/final_checkpoint_seed{self.random_seed}.pth'
                )

        if self.use_early_stopping:
            # Load the best model
            assert os_path.exists(self.tmp_dir), 'Tmp dir does not exist.'
            print(f'Loading best checkpoint from epoch {self.early_stopping.best_epoch}...')
            checkpoint = torch.load(self.tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint

            # Remove the early stopping checkpoint
            os_remove(self.tmp_dir + f'/best_checkpoint_seed{self.random_seed}.pth')

        if self.device.type == 'cuda':
            del train_data_loader, val_data_loader, train_iterator, val_iterator
            torch.cuda.empty_cache()

    def predict(
        self,
        test_data_loader
        ):

        self.model.eval()
        attn_weights_list = []
        test_probs = []
        test_preds = []
        test_sample_keys = []
        test_labels = []
        with torch.no_grad():
            for seq_batch, label_batch, mask_batch, sample_key_batch in test_data_loader:
                probs, preds, _, attn_weights = self.evaluate_batch(
                    seq_batch=seq_batch,
                    label_batch=None,
                    mask_batch=mask_batch,
                    mode='test',
                    criterion=None,
                    chunk_size=self.chunk_size
                )
                if attn_weights is not None and self.store_attention_weights:
                    attn_weights_list.append(attn_weights)
                test_probs.append(probs)
                test_preds.append(preds)
                sample_key_batch = sample_key_batch[::self.n_spatial_channels * self.n_temporal_channels]  # Take every n_spatial * n_temporal key because they are repeated (n_spatial * n_temporal) times
                test_sample_keys.extend(sample_key_batch)
                test_labels.extend(label_batch.numpy())

        if len(attn_weights_list) != 0:
            attn_weights = torch.cat(attn_weights_list).squeeze().cpu()
            attn_weights_dict = {
                sample_id: attn_weights[i] for i, sample_id in enumerate(test_sample_keys)
            }
        else:
            attn_weights_dict = None

        if self.problem_type == 'binary':
            prob_df = pd.DataFrame({'prob': torch.cat(test_probs).cpu().numpy()})
            prob_df.index = test_sample_keys
            prob_df.columns = ['prob']
        elif self.problem_type == 'multiclass':
            prob_df= pd.DataFrame(torch.cat(test_probs).cpu().numpy())
            prob_df.index = test_sample_keys
            prob_df.columns = [f'prob_class_{i}' for i in range(prob_df.shape[1])]
        else:
            prob_df = None

        pred_df = pd.DataFrame({'pred': torch.cat(test_preds).cpu().numpy()})
        pred_df.index = test_sample_keys
        pred_df.columns = ['pred']

        extra_info = {
            'best_epoch': self.early_stopping.best_epoch if self.use_early_stopping else None,
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
            'attn_weights': attn_weights_dict,
            'prob_df': prob_df,
            'test_labels': test_labels
        }

        return pred_df, extra_info

    def initialize_optimizer(self):
        optimizer_name = self.optimizer_params['optimizer_name']

        if optimizer_name == 'adam':
            self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_params['initial_lr'])
        elif optimizer_name == 'adamw':
            self.model.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr_params['initial_lr'],
                betas=self.optimizer_params.get('betas', (0.9, 0.999)),
                eps=self.optimizer_params.get('eps', 1e-8),
                weight_decay=self.optimizer_params.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f'Given optimizer name, {optimizer_name}, is not valid. Valid names are adam and adamw.')

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

    def initialize_train_val_iterators(self, train_data_loader, val_data_loader):
        if self.use_tqdm:
            train_iterator = tqdm(train_data_loader, 'Training batches...')
            val_iterator = tqdm(val_data_loader, 'Validation batches...')
        else:
            train_iterator = train_data_loader
            val_iterator = val_data_loader

        return train_iterator, val_iterator

    def evaluate_batch(
        self,
        seq_batch,
        label_batch,
        mask_batch,
        mode,
        criterion,
        chunk_size
        ):
        # Move tensors to the specified device
        seq_batch = seq_batch.to(self.device) # Shape: (batch_size, max_hr, n_channels, n_features)
        mask_batch = mask_batch.to(self.device) # Shape: (batch_size, max_hr, n_channels, n_features)

        if mode == 'train':
            self.model.optimizer.zero_grad()
        return_attention = (mode == 'test' and self.store_attention_weights)
        with torch.amp.autocast('cuda', enabled=True):
            logits, attn_weights = self.model(x=seq_batch, mask=mask_batch, return_attention=return_attention, chunk_size=chunk_size) # Binary shape: (batch_size, 1), Multiclass shape: (batch_size, n_classes)
        if self.problem_type == 'binary':
            logits = logits.squeeze()  # Remove class dimension for binary

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

        return probs, preds, loss, attn_weights

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

    def save_checkpoint(self, epoch, train_loss, val_loss, checkpoint_path, is_final=False):
        """Save model checkpoint with training state"""
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'train_main_loss': train_loss,
            'val_main_loss': val_loss,
            'training_history': {
                'train_main_losses': self.train_main_losses,
                'val_main_losses': self.val_main_losses,
                'train_balanced_acc_list': self.train_balanced_acc_list,
                'val_balanced_acc_list': self.val_balanced_acc_list,
                'train_roc_auc_list': getattr(self, 'train_roc_auc_list', []),
                'val_roc_auc_list': getattr(self, 'val_roc_auc_list', []),
                'train_pr_auc_list': getattr(self, 'train_pr_auc_list', []),
                'val_pr_auc_list': getattr(self, 'val_pr_auc_list', []),
                'train_cohen_kappa_list': getattr(self, 'train_cohen_kappa_list', []),
                'val_cohen_kappa_list': getattr(self, 'val_cohen_kappa_list', []),
                'train_weighted_f1_list': getattr(self, 'train_weighted_f1_list', []),
                'val_weighted_f1_list': getattr(self, 'val_weighted_f1_list', []),
                'train_cm_list': getattr(self, 'train_cm_list', []),
                'val_cm_list': getattr(self, 'val_cm_list', [])
            },
            'random_seed': self.random_seed
        }
        
        # Add scheduler state if using scheduler
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")