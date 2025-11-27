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


import os
from pathlib import Path
import numpy as np
import pandas as pd

SEG_LABEL_COLS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
SEG_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
CLASSIFICATION_DATA_DIR = '/home/tibia/Projet_Hemorragie/MBH_label_case'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_multitask_libMTL2/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

set.device('cuda' if torch.cuda.is_available() else 'cpu')
#on utilise le GPU 1 
set.cuda_device(1)
# ======================
# DATA PREPARATION
# ======================
def get_segmentation_data(split="train"):
    img_dir = Path(SEG_DIR) / split / "img"
    seg_dir = Path(SEG_DIR) / split / "seg"
    
    images = sorted(img_dir.glob("*.nii.gz"))
    labels = sorted(seg_dir.glob("*.nii.gz"))
    
    assert len(images) == len(labels), "Mismatch between image and label counts"

    data = []
    for img, lbl in zip(images, labels):
        data.append({
            "image": str(img),
            "label": str(lbl),
        })
        
    return data


def get_classification_data(split="train"):
    csv_path = Path(CLASSIFICATION_DATA_DIR) / "splits" / f"{split}_split.csv"
    df = pd.read_csv(csv_path)
    nii_dir = Path(CLASSIFICATION_DATA_DIR)
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    data = []
    for _, row in df.iterrows():
        image_path = str(nii_dir / f"{row['patientID_studyID']}.nii.gz")
        label = np.array([row[col] for col in label_cols], dtype=np.float32)
        
        data.append({
            "image": image_path,
            "label": label
        })
    return data
#Définitions des transformations MONAI
#Définitions des transformations MONAI
from monai import transforms as T
import torch
 
def get_segmentation_transform(mode='train'):
    # Transforms de base (toujours appliquées)
    base_transforms = [
        T.LoadImaged(keys=["image", "label"], image_only=True ),
        T.EnsureChannelFirstd(keys=["image", "label"]),
        T.CropForegroundd(keys=["image", "label"], source_key='image'),
        T.Orientationd(keys=["image", "label"], axcodes='RAS'),
        T.Spacingd(keys=["image", "label"], pixdim=(1., 1., 1.), mode=["bilinear", "nearest"]),
        T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        T.ScaleIntensityRanged(
            keys=["image"],
            a_min=-10,
            a_max=140,
            b_min=0.0, b_max=1.0, clip=True
        ),
        T.SelectItemsD(keys=["image", "label"])
        
    ]
    augmentation_transforms = []
    if mode == 'train':
        augmentation_transforms = [
            T.RandCropByPosNegLabeld(
                keys=['image', 'label'],
                image_key='image',
                label_key='label',
                pos=5.0,
                neg=1.0,
                spatial_size=(96, 96, 96),
                num_samples=2
            ),
            T.RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5),
            T.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            
        ]
        
        
        # T.RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        
        # # 3. Optionnel mais top : Flou (simulation de mouvement patient) ou Netteté
        # T.RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0), prob=0.1),
     # à tester
        
    final_transform = [T.EnsureTyped(keys=["image", "label"], track_meta=False)]
    

    all_transforms = base_transforms + augmentation_transforms + final_transform
    
       
    
    return T.Compose(all_transforms)
    
    

    


def get_classification_transform(mode='train'):
    # Transforms de base (toujours appliquées)
    base_transforms = [
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.CropForegroundd(keys=["image"], source_key='image'),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,a_max=140, 
                b_min=0.0, b_max=1.0, 
                clip=True) ,
            T.RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False)]
        
    augmentation_transforms = []      
    if mode == 'train':
        augmentation_transforms = [
            T.RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
            T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            
        ]
        
    final_transform = [T.ToTensord(keys=["image", "label"]),
                       T.SelectItemsD(keys=["image", "label"]),
                       T.EnsureTyped(keys=["image", "label"], track_meta=False)]
        
    all_transforms = base_transforms + augmentation_transforms + final_transform
        
    return T.Compose(all_transforms)
# Préparation dataloaders
from monai.data import DataLoader, PersistentDataset
seg_train_data=get_segmentation_data("train")
cls_train_data=get_classification_data("train")

seg_train_dataset = PersistentDataset(
        seg_train_data, 
        transform=get_segmentation_transform('train'),
        cache_dir=os.path.join(SAVE_DIR, "cache_train")
    )

cls_train_dataset = PersistentDataset(
        cls_train_data,
        transform=get_classification_transform('train'),
        cache_dir=os.path.join(SAVE_DIR, "cache_train"))
    
#Val dataset
seg_val_data=get_segmentation_data("val")
cls_val_data=get_classification_data("val")
seg_val_dataset = PersistentDataset(
        seg_val_data, 
        transform=get_segmentation_transform('val'),    
        cache_dir=os.path.join(SAVE_DIR, "cache_val")
    )   
cls_val_dataset = PersistentDataset(
        cls_val_data,
        transform=get_classification_transform('val'),
        cache_dir=os.path.join(SAVE_DIR, "cache_val"))      


# DataLoaders
seg_train_loader = DataLoader(
        seg_train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
)

cls_train_loader = DataLoader(
        cls_train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
)
  
train_dataloaders = {'segmentation': seg_train_loader,
                     'classification': cls_train_loader
                     }


seg_val_loader = DataLoader(
        seg_val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True,
)   

cls_val_loader = DataLoader(  
        cls_val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True,
)
val_dataloaders = {'segmentation': seg_val_loader,
                   'classification': cls_val_loader
                   }


# ======================
# MÉTRIQUES PERSONNALISÉES
# ======================

# Losses
# Ponderer ensuite pa classe avec WeightSampler

from LibMTL.loss import AbsLoss
import torch
from monai.losses import DiceCELoss

class ClassificationLossWrapper(AbsLoss):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt.float())
    
class SegmentationLossWrapper(AbsLoss):
    def __init__(self):
        super().__init__()
        self.loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True
        )

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)  
    
    
    from LibMTL.metrics import AbsMetric
import torch

from torchmetrics.classification import MultilabelAUROC

class MultiLabelAUCMetric(AbsMetric):
    def __init__(self, num_labels=6):
        super().__init__()
        self.metric = MultilabelAUROC(num_labels=num_labels, average=None)   # par classe
        self.metric_mean = MultilabelAUROC(num_labels=num_labels, average="macro")  # moyenne
        self.num_labels = num_labels

    def update_fun(self, pred, gt):
        # pred = logits -> transform needed
        pred = torch.sigmoid(pred)
        gt=gt.detach().cpu().long() #  pour torchmetrics veut des long
        pred=pred.detach().cpu()
        
        self.metric.update(pred,gt)
        self.metric_mean.update(pred,gt)

    def score_fun(self):
        per_class = self.metric.compute().tolist()
        mean_auc = self.metric_mean.compute().item()
        return per_class + [mean_auc]

    def reinit(self):
        super().reinit()
        self.metric.reset()
        self.metric_mean.reset() 
        
        
# Loss
from LibMTL.metrics import AbsMetric
from monai.metrics import DiceMetric,DiceHelper
from monai.utils import MetricReduction, deprecated_arg
        
class DiceMetricAdapter(AbsMetric):
    """
    Cet adaptateur implémente AbsMetric pour calculer le Dice Score correctement.
    
    - `update_fun` utilise DiceHelper pour obtenir les scores bruts (B, C) 
      et les stocke dans `self.record`.
    - `score_fun` agrège tous les scores de `self.record` et calcule 
      la moyenne finale (le "score des totaux" émulé).
    """
    def __init__(self, num_classes, include_background=False):
        # Initialise self.record et self.bs
        super().__init__()
        
        self.num_classes = num_classes
        self.include_background = include_background
        
        # On utilise DiceHelper comme "calculateur" ponctuel.
        # On lui demande de NE PAS faire de réduction (reduction="none")
        # car on veut stocker les scores bruts (Batch, Classes).
        self.dice_helper = DiceHelper(
            include_background=include_background,
            num_classes=num_classes,
            reduction=MetricReduction.NONE,
            ignore_empty=True,  # Important : ignore les cas où le GT est vide
            apply_argmax=False  # On le fera nous-mêmes dans update_fun
        )

    def update_fun(self, pred, gt):
        """
        Appelé à chaque batch. Calcule les scores (B, C) et les stocke.
        
        Args:
            pred (torch.Tensor): Prédictions (logits) de forme (B, C, H, W, D)
            gt (torch.Tensor): Vérité terrain (labels) de forme (B, 1, H, W, D)
        """
        # 1. Convertir les logits en labels
        # DiceHelper attend des labels, pas des logits
        pred_labels = torch.argmax(pred, dim=1, keepdim=True)
        
        # 2. Calculer les scores Dice pour ce batch
        # Le résultat est un tenseur de (B, num_classes_calculées)
        # ex: (B, 5) si num_classes=6 et include_background=False
        batch_dice_scores,_ = self.dice_helper(pred_labels, gt)
        
        # 3. Stocker ce tenseur dans notre "record"
        self.record.append(batch_dice_scores)
        
        # 4. Stocker la taille du batch (comme le fait AbsMetric)
        self.bs.append(pred.shape[0])

    def score_fun(self):
        """
        Appelé à la fin de l'époque. Agrège les scores et calcule la moyenne. Peut etre à modifier pour le loggage de chaque dice
        """
        if not self.record:
            # Retourne un score pour chaque classe, mis à 0
            num_expected_classes = self.num_classes - (1 if not self.include_background else 0)
            return torch.zeros(num_expected_classes)
            
        # 1. Rassembler tous les tenseurs de (B, C) en un seul
        # grand tenseur de (Total_B, C)
        all_scores = torch.cat(self.record, dim=0)
        
        # 2. Calculer la moyenne sur la dimension des batches (dim=0)
        # On utilise nanmean pour ignorer les NaN (cas des GT vides)
        # C'est la façon correcte d'agréger le Dice.
        mean_scores_per_class = torch.nanmean(all_scores, dim=0)
        #mean_gloabal = torch.nanmean(mean_scores_per_class)
        
        # `score_fun` est censé retourner une "liste", mais un tenseur
        # est plus utile. On retourne la moyenne par classe.
        return mean_scores_per_class.tolist()
    
    # La méthode reinit() est héritée de AbsMetric et fonctionne parfaitement
    # car elle vide self.record et self.bs.
     

# dictionnaire de tâches
class_names = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
seg_metric_names_list= [ 'EDH', ]
metric_names_list = [f"AUC_{name}" for name in class_names] + ["AUC_Mean"]
seg_metric_names_list = [f"Dice_{name}" for name in class_names]

print("Noms des métriques de classification :", metric_names_list)
print("Noms des métriques de segmentation   :", seg_metric_names_list)
# dictionnaire de tâches

task_dict = {
    'classification': {
        'loss_fn': ClassificationLossWrapper(),
        'metrics_fn': MultiLabelAUCMetric(num_labels=6),
        'metrics': ['val_auc_class_0', 'val_auc_class_1', 'val_auc_class_2', 
                   'val_auc_class_3', 'val_auc_class_4', 'val_auc_class_5', 
                   'val_auc_mean'],
        'weight': [1.0]
    },
    'segmentation': {
        'loss_fn': SegmentationLossWrapper(),
        'metrics_fn': DiceMetricAdapter(num_classes=6, include_background=False),
        'metrics': ['dice_c1', 'dice_c2', 'dice_c3', 'dice_c4', 'dice_c5'],
        'weight': [1.0] * 5
    }
}
# self.task_num = len(task_dict)
task_num = len(task_dict)
print(f"Nombre de tâches définies : {task_num}")

# ======================
#Def modèle Trainer LibMTL
# ======================
from typing import Sequence
import torch
import torch.nn as nn
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat


class HemorrhageEncoder(nn.Module):
    """
    Cette classe contient la partie descendante (encodeur) du U-Net.
    Elle est partagée par les deux tâches.
    Son forward pass retourne une liste de toutes les feature maps
    nécessaires pour les skip connections du décodeur de segmentation.
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
    ):
        super().__init__()
        
        # Assure que 'features' a la bonne longueur
        self.fea = nn.Parameter(torch.tensor(features), requires_grad=False)
        
        self.conv_0 = TwoConv(spatial_dims, in_channels, self.fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, self.fea[0], self.fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, self.fea[1], self.fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, self.fea[2], self.fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, self.fea[3], self.fea[4], act, norm, bias, dropout)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Le forward pass de l'encodeur.
        Retourne une liste contenant le bottleneck (x4) et toutes les
        sorties intermédiaires pour les skip connections.
        """
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)  # C'est le bottleneck (la représentation partagée)
        
        return [x4, x3, x2, x1, x0]

# ========================================================================
# 2. LES DÉCODEURS (Les têtes spécifiques à chaque tâche)
# ========================================================================

class SegmentationDecoder(nn.Module):
    """
    Le décodeur pour la tâche de segmentation.
    Il prend la liste de features de l'encodeur et reconstruit le masque.
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        out_channels: int = 6,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        
        fea = nn.Parameter(torch.tensor(features), requires_grad=False)
        
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv = nn.Conv3d(fea[5], out_channels, kernel_size=1)

    def forward(self, enc_out: list[torch.Tensor]) -> torch.Tensor:
        # On récupère les tenseurs de la liste fournie par l'encodeur
        x4, x3, x2, x1, x0 = enc_out
        
        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        
        return self.final_conv(u1)

class ClassificationDecoder(nn.Module):
    """
    Le décodeur pour la tâche de classification.
    Il prend la liste de features de l'encodeur mais n'utilise que le
    bottleneck (x4) pour prédire les classes.
    """
    def __init__(
        self,
        in_features: int,  # Doit correspondre à features[4] de l'encodeur
        num_cls_classes: int = 6,
    ):
        super().__init__()
        
        # Tête de classification, exactement comme avant
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(in_features * 4 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_cls_classes)
        )
        

    def forward(self, enc_out: list[torch.Tensor]) -> torch.Tensor:
        # On ne prend que le bottleneck (le premier élément de la liste)
        x4 = enc_out[0]
        
        # Toute la logique d'agrégation de patches a disparu !
        # On passe directement les features à la tête de classification.
        return self.cls_head(x4)

# ========================================================================
# 3. ASSEMBLAGE FINAL POUR LibMTL
# ========================================================================

# Définis tes paramètres
task_name = ["segmentation", "classification"]
features = (32, 32, 64, 128, 256, 32)

# Crée une instance de l'encodeur partagé
encoder = HemorrhageEncoder(features=features)

# Crée un dictionnaire de décodeurs
decoders = nn.ModuleDict({
    'segmentation': SegmentationDecoder(
        out_channels=6, # 6 classes de segmentation
        features=features
    ),
    'classification': ClassificationDecoder(
        in_features=features[4], # La taille du bottleneck (256)
        num_cls_classes=6 # 6 classes de classification
    )
})





# ========================================================================
# 3. ASSEMBLAGE FINAL POUR LibMTL
# ========================================================================
# Paramètres optim & scheduler
optim_param = {
    'optim': 'sgd', 
    'lr': 1e-3, 
    'weight_decay': 3e-5,  # 0.00003 est égal à 3e-5
    'momentum': 0.99, 
    'nesterov': True
}

lengths = [len(loader) for loader in train_dataloaders.values()]

# 2. Trouver le dataloader le plus long (c'est sur lui que LibMTL se cale)
steps_per_epoch = max(lengths)


total_steps = steps_per_epoch * 1000  # 1000 epochs 

scheduler_param = {
    'scheduler': 'linearschedulewithwarmup',  # Correspond à get_linear_schedule_with_warmup
    'num_warmup_steps': 0, 
    'num_training_steps': total_steps
   }
# --- 2. DÉFINITION MANUELLE DE KWARGS  ---

# Arguments spécifiques à l'architecture (Exemple pour un U-Net 3D ou une archi complexe)
arch_args = {
    
    # Si vous utilisez CGC, PLE, ou MMoE, vous devez spécifier la taille d'image et le nombre d'experts
    # 'img_size': (96, 96, 96), 
    # 'num_experts': [4, 4, 4], 
    
    # Si votre encodeur ResNet a des arguments spécifiques, mettez-les ici
    # Ex: 'channels': 3 # Si vous devez le passer à l'initialisation de l'encodeur
}

# Arguments spécifiques à la méthode de pondération (Exemple pour 'EW' qui n'a besoin de rien)
weight_args = {
    # Pour EW (Equal Weighting), c'est souvent vide.
}

# Si vous utilisiez DWA, vous définiriez T :
# weight_args = {'T': 1.0} 

# Si vous utilisiez GradNorm, vous définiriez alpha :
# weight_args = {'alpha': 0.1}

# --- 3. CONSOLIDER KWARGS (Optionnel mais propre) ---

# Crée le dictionnaire kwargs global
kwargs = {
    'arch_args': arch_args,
    'weight_args': weight_args
}



import wandb

config_l = dict(
    sharing_type="hard",   # "soft" ou "fine_tune"
    model="BasicUNetWithClassification",
    loss_weighting="none",
    dataset_size="balanced",  # "full" ou "balanced" ou "optimized"
    batch_size=2,
    learning_rate=1e-3,
    optimizer="sgd",
    batch_stratégie= "loop", 
    seed=42
)
torch.cuda.set_device(0)
# Génération automatique de tags à partir de config
tags = [f"{k}:{v}" for k, v in config_l.items() if k in ["sharing_type", "optimizer", "model", "loss_weighting"]]


# : Initialisation manuelle de wandb
# Au lieu de : wandb_logger = WandbLogger(...)
wandb.init(
    project="hemorrhage_multitask_test",
    group="noponderation",
    tags=tags,
    config=config_l,
    name="multitask_unet3d_libMTL"
)




# 3️ Méthodes multitâches
from LibMTL.architecture import HPS
from LibMTL.weighting import GradNorm

# 4️ Instanciation du Trainer
from LibMTL.trainer import Trainer

hemorrhage_trainer = Trainer(
    task_dict=task_dict,
    weighting= 'EW',
    architecture='Unet_hemo',
    #save_path=SAVE_DIR, à ajouter 
    encoder_class=HemorrhageEncoder,
    decoders=decoders,
    rep_grad=False,
    multi_input=True,
    optim_param=optim_param,
    scheduler_param=scheduler_param,
    #device='cuda',
    **kwargs
)

#  Entraînement
hemorrhage_trainer.train(train_dataloaders, test_dataloaders = None, epochs=1000 , val_dataloaders=val_dataloaders)
