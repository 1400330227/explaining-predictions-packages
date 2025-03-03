import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.models import resnet50
from sklearn import metrics


def compute_accuracy(labels, preds):
    accuracy_score = metrics.accuracy_score(labels, preds)
    precision_score = metrics.precision_score(labels, preds, average='macro')
    recall_score = metrics.recall_score(labels, preds, average='macro')
    f1_score = metrics.f1_score(labels, preds, average='macro')
    return accuracy_score, precision_score, recall_score, f1_score


def to_np(x):
    return x.data.cpu().numpy()


class ConceptEmbeddingModel(pl.LightningModule):
    def __init__(self, n_concepts, embedding_activation, emb_size, n_tasks, c2y_model, weight_loss, task_class_weights,
                 concept_loss_weight):
        super().__init__()
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.concept_loss_weight = concept_loss_weight
        self.pre_concept_model = resnet50(pretrained=True)
        self.emb_size = emb_size
        self.n_concepts = n_concepts

        for i in range(n_concepts):
            self.concept_context_generators.append(
                torch.nn.Sequential(*[
                    torch.nn.Linear(list(self.pre_concept_model.modules())[-1].out_features, 2 * emb_size),
                    torch.nn.LeakyReLU(),
                ])
            )
            self.concept_prob_generators.append(
                torch.nn.Sequential(*[
                    torch.nn.Linear(2 * emb_size, 1),
                ])
            )
        if c2y_model is None:
            units = [n_concepts * emb_size] + [n_tasks]
            layers = []
            layers.append(torch.nn.Linear(units[0], units[1]))
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()
        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights) if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights))

        self.shared_prob_gen = True

    def forward(self, x, intervention_idxs=None, c=None, y=None, train=False, latent=None, competencies=None,
                prev_interventions=None):
        pre_c = self.pre_concept_model(x)
        contexts = []
        c_sem = []

        for i, context_gen in enumerate(self.concept_context_generators):
            prob_gen = self.concept_prob_generators[0] if self.shared_prob_gen else self.concept_prob_generators[i]
            context = context_gen(pre_c)  # 1000->32
            prob = prob_gen(context)  # 32->1

            contexts.append(torch.unsqueeze(context, dim=1))
            c_sem.append(self.sig(prob))
        c_sem = torch.cat(c_sem, dim=-1)  # batch*112
        contexts = torch.cat(contexts, dim=1)  # batch*112*32
        latent = contexts, c_sem
        probs = c_sem

        c_pred = (contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
                  contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1)))

        c_pred = c_pred.reshape(-1, self.emb_size * self.n_concepts)
        y = self.c2y_model(c_pred)
        return tuple([c_sem, c_pred, y])

    def training_step(self, batch, batch_idx):
        loss, result = self._run_step(batch, batch_idx, train=True, intervention_idxs=None)
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result['c_accuracy'],
                "c_auc": result['c_precision'],
                "c_f1": result['c_f1'],
                "y_accuracy": result['y_accuracy'],
                "y_auc": result['y_precision'],
                "y_f1": result['y_f1'],
                "concept_loss": result['concept_loss'],
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "avg_c_y_acc": result['avg_c_y_acc'],
            },
        }

    def validation_step(self, batch, batch_idx):
        _, result = self._run_step(batch, batch_idx, train=False)
        return result

    def testing_step(self, batch, batch_idx):
        _, result = self._run_step(batch, batch_idx, train=False)
        return result['loss']

    def _run_step(self, batch, batch_idx, train=False, intervention_idxs=None):
        x, y, c = batch[0], batch[1], batch[2]
        competencies, prev_interventions = None, None
        outputs = self.forward(x, intervention_idxs=intervention_idxs, c=c, y=y, train=train, competencies=competencies,
                               prev_interventions=prev_interventions)
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        task_loss = self.loss_task(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)
        task_loss_scalar = task_loss.detach()
        concept_loss = self.loss_concept(c_sem, c)
        concept_loss_scalar = concept_loss.detach()

        loss = self.concept_loss_weight * concept_loss + task_loss

        c_pred = (c_sem.cpu().detach().numpy() >= 0.5).astype(np.int32)
        c_true = (c.cpu().detach().numpy() > 0.5).astype(np.int32)
        c_accuracy, c_precision, c_recall, c_f1 = compute_accuracy(c_true, c_pred)

        y_pred = y_logits.argmax(dim=-1).cpu().detach().numpy()
        y_accuracy, y_precision, y_recall, y_f1 = compute_accuracy(to_np(y), y_pred)

        result = {
            'c_accuracy': c_accuracy,
            'c_precision': c_precision,
            'c_recall': c_recall,
            'c_f1': c_f1,
            'y_accuracy': y_accuracy,
            'y_precision': y_precision,
            'y_recall': y_recall,
            'y_f1': y_f1,
            'concept_loss': concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }

        print(result)

        return loss, result

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.01, momentum=0.9,
                                    weight_decay=5e-4)
        # 达到一个 “平台期” 的情况时，降低学习率可以帮助模型跳出局部最优解，继续朝着全局最优解前进
        lr_shceduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_shceduler,
            'monitor': 'loss',
        }
