import LibMTL
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from LibMTL import Trainer
from LibMTL.loss import *
from LibMTL.metrics import *
import monai
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

# Configuration
SEG_LABEL_COLS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
SEG_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
CLASSIFICATION_DATA_DIR = '/home/tibia/Projet_Hemorragie/MBH_label_case'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_multitask_pos_cases"

# ======================
# DÉFINITION DES TÂCHES
# ======================

task_dict = {
    'segmentation': {
        'metrics': ['dice'],  # Nous allons créer une métrique Dice personnalisée
        'metrics_fn': None,   # Défini plus tard
        'loss_fn': None,      # Défini plus tard
        'weight': 1.0         # Poids pour la loss
    },
    'classification': {
        'metrics': ['auroc'],
        'metrics_fn': None,
        'loss_fn': None,
        'weight': 1.0
    }
}

# ======================
# MÉTRIQUES PERSONNALISÉES
# ======================

class DiceMetricCustom:
    def __init__(self, num_classes=len(SEG_LABEL_COLS)):
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.num_classes = num_classes
    
    def __call__(self, pred, target):
        # pred: [B, C, H, W, D], target: [B, C, H, W, D] (one-hot)
        return self.dice_metric(pred, target).mean()

class MultilabelAUROC:
    def __init__(self, num_classes=len(SEG_LABEL_COLS)):
        self.num_classes = num_classes
    
    def __call__(self, pred, target):
        # Implémentation simplifiée - vous pouvez utiliser celle de MONAI
        from sklearn.metrics import roc_auc_score
        pred_np = torch.sigmoid(pred).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        try:
            auroc = roc_auc_score(target_np, pred_np, average='macro')
            return torch.tensor(auroc)
        except:
            return torch.tensor(0.5)

# Initialisation des métriques
task_dict['segmentation']['metrics_fn'] = DiceMetricCustom()
task_dict['classification']['metrics_fn'] = MultilabelAUROC()

# ======================
# MODÈLE MTL
# ======================

class HemorrhageMTLModel(LibMTL.model.MTLModel):
    def __init__(self, task_dict, **kwargs):
        super(HemorrhageMTLModel, self).__init__(task_dict, **kwargs)
        
        # Encodeur CNN commun (comme dans votre approche actuelle)
        self.shared_encoder = monai.networks.nets.DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=512  # Features de sortie
        )
        
        # Tête pour la segmentation
        self.segmentation_head = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, len(SEG_LABEL_COLS), 1)  # Classes de segmentation
        )
        
        # Tête pour la classification
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(SEG_LABEL_COLS))  # Classes de classification
        )
    
    def forward(self, inputs, task_name):
        # Passage par l'encodeur commun
        features = self.shared_encoder(inputs)
        
        if task_name == 'segmentation':
            return self.segmentation_head(features)
        elif task_name == 'classification':
            return self.classification_head(features)

# ======================
# DATASETS ET DATALOADERS
# ======================

class HemorrhageDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.seg_data = self.get_segmentation_data()
        self.cls_data = self.get_classification_data()
        
    def get_segmentation_data(self):
        img_dir = Path(SEG_DIR) / self.split / "img"
        seg_dir = Path(SEG_DIR) / self.split / "seg"
        
        images = sorted(img_dir.glob("*.nii.gz"))
        labels = sorted(seg_dir.glob("*.nii.gz"))
        
        data = []
        for img, lbl in zip(images, labels):
            data.append({
                "image": str(img),
                "label": str(lbl),
                "task": "segmentation"
            })
        return data
    
    def get_classification_data(self):
        csv_path = Path(CLASSIFICATION_DATA_DIR) / "splits" / f"{self.split}_split.csv"
        df = pd.read_csv(csv_path)
        nii_dir = Path(CLASSIFICATION_DATA_DIR)
        
        data = []
        for _, row in df.iterrows():
            image_path = str(nii_dir / f"{row['patientID_studyID']}.nii.gz")
            label = np.array([row[col] for col in SEG_LABEL_COLS], dtype=np.float32)
            
            data.append({
                "image": image_path,
                "label": label,
                "task": "classification"
            })
        return data
    
    def __len__(self):
        return len(self.seg_data) + len(self.cls_data)
    
    def __getitem__(self, idx):
        if idx < len(self.seg_data):
            item = self.seg_data[idx]
            task = "segmentation"
        else:
            item = self.cls_data[idx - len(self.seg_data)]
            task = "classification"
        
        # Ici vous ajouterez le chargement des images NIfTI
        # et le prétraitement avec MONAI
        return {
            'input': item['image'],  # À adapter avec le vrai chargement
            'label': item['label'],
            'task': task
        }

# ======================
# CONFIGURATION D'ENTRAÎNEMENT
# ======================

def main():
    # Définition des losses
    seg_loss_fn = DiceCELoss(include_background=False)
    cls_loss_fn = torch.nn.BCEWithLogitsLoss()
    
    task_dict['segmentation']['loss_fn'] = seg_loss_fn
    task_dict['classification']['loss_fn'] = cls_loss_fn
    
    # Configuration LibMTL
    config = {
        'multi_input': True,  # Chaque tâche a son propre dataset
        'task_dict': task_dict,
        'optimizer': 'adam',
        'optimizer_params': {'lr': 1e-4},
        'scheduler': 'step',
        'scheduler_params': {'step_size': 10, 'gamma': 0.1}
    }
    
    # Création du modèle
    model = HemorrhageMTLModel(task_dict=task_dict)
    
    # Datasets
    train_dataset = HemorrhageDataset(split="train")
    val_dataset = HemorrhageDataset(split="val")
    
    # Trainer LibMTL
    trainer = Trainer(
        task_dict=task_dict,
        weighting='equal',  # Méthode de pondération la plus simple
        architecture='hps',  # Hard Parameter Sharing (le plus simple)
        encoder_class=model.shared_encoder,
        decoders=model.task_heads,
        rep_grad=False
    )
    
    # Entraînement (pseudocode - à adapter)
    # trainer.run(train_dataloaders, val_dataloaders, ...)

if __name__ == "__main__":
    main()