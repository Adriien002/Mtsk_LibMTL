from LibMTL.architecture.abstract_arch import AbsArchitecture

class Unet_hemo(AbsArchitecture):
    """
    Architecture personnalisée pour U-Net Multitâche.
    Elle envoie TOUTES les features (la liste complète)
    à chaque décodeur, au lieu de les découper par tâche.
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        # 1. On initialise la classe parente (qui stocke decoders, device, etc.)
        super(Unet_hemo, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class(**kwargs)
        
        
        
    def forward(self, inputs, task_name=None):
        out = {}
        
        # 1. On récupère la liste complète des features du U-Net
        # s_rep est une liste : [x0, x1, x2, x3, x4]
        s_rep = self.encoder(inputs) 
        
        # 2. On distribue la liste COMPLÈTE à chaque décodeur
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            
            if self.rep_grad:
                # Cette fonction remplit self.rep[task] utilisé par GradNorm.
                # On met same_rep=False pour que chaque tâche ait son entrée dans le dictionnaire.
                # On ignore la valeur de retour (qui est détachée) car on veut utiliser la vraie liste pour le décodeur.
                self._prepare_rep(s_rep[0], task, same_rep=False) # on prend le premier élement de la liste qui correspond au bottleneck
            
            
            # On ne fait PAS s_rep[tn]. On donne tout 's_rep'.
            # C'est le décodeur qui se débrouillera pour prendre ce qu'il veut.
            out[task] = self.decoders[task](s_rep)
            
        return out