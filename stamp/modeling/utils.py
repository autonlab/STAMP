# Load model directly
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score
from types import SimpleNamespace
from CBraMod.models import (model_for_bciciv2a, model_for_chb, model_for_physio, 
                                model_for_mumtaz, model_for_faced, model_for_isruc, model_for_seedv, model_for_seedvig,
                                model_for_shu, model_for_speech, model_for_stress, model_for_tuab, model_for_tuev)

def calculate_binary_performance_metrics(truths, probs, preds):
    balanced_acc = balanced_accuracy_score(truths, preds)
    roc_auc = roc_auc_score(truths, probs)
    precision, recall, _ = precision_recall_curve(truths, probs, pos_label=1)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(truths, preds)

    return balanced_acc, pr_auc, roc_auc, cm

def calculate_multiclass_performance_metrics(truths, preds):
    # Convert y_true to int if it is float
    if 'float' in str(preds.dtype):
        preds = preds.astype(int)

    balanced_acc = balanced_accuracy_score(truths, preds)
    cohen_kappa = cohen_kappa_score(truths, preds)
    weighted_f1 = f1_score(truths, preds, average='weighted')
    cm = confusion_matrix(truths, preds)

    return balanced_acc, cohen_kappa, weighted_f1, cm

def get_cbramod_model(model_params, dataset_name):

    # Convert model_params to SimpleNamespace
    model_params = SimpleNamespace(**model_params)
    if dataset_name == 'bciciv2a':
        model = model_for_bciciv2a.Model(model_params)
    elif dataset_name == 'chb':
        model = model_for_chb.Model(model_params)
    elif dataset_name == 'physio':
        model = model_for_physio.Model(model_params)
    elif dataset_name == 'mumtaz':
        model = model_for_mumtaz.Model(model_params)
    elif dataset_name == 'faced':
        model = model_for_faced.Model(model_params)
    elif dataset_name == 'isruc':
        model = model_for_isruc.Model(model_params)
    elif dataset_name == 'seedv':
        model = model_for_seedv.Model(model_params)
    elif dataset_name == 'seedvig':
        model = model_for_seedvig.Model(model_params)
    elif dataset_name == 'shu':
        model = model_for_shu.Model(model_params)
    elif dataset_name == 'speech':
        model = model_for_speech.Model(model_params)
    elif dataset_name == 'stress':
        model = model_for_stress.Model(model_params)
    elif dataset_name == 'tuab':
        model = model_for_tuab.Model(model_params)
    elif dataset_name == 'tuev':
        model = model_for_tuev.Model(model_params)
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')

    return model

