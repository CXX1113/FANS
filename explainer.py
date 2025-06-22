import os
from math import sqrt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device, curr_dir
from utils import read_yaml, GaussianKernel, AitchisonAitkenKernel, resampling, build_sample_data, minmax_norm
from utils import compute_class_bandwidths, get_temperature, concrete_sample, perturb_func


# X->M
class FeatureSelector(nn.Module):
    def __init__(self, target_image, init_method, init_mask, feature_is_continuous, num_levels=2):
        super().__init__()
        self.target_image = target_image
        _, num_channel, width, height = self.target_image.shape

        if init_mask is not None:
            self.feat_mask = nn.Parameter(init_mask.repeat(num_channel, 1, 1))
        elif init_method == 'gnn_explainer':
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * width))
            self.feat_mask = nn.Parameter(torch.randn(1, width, height, device=device) * std)
        elif init_method == 'normal':
            self.feat_mask = nn.Parameter(torch.randn(1, width, height, device=device))
        elif init_method == 'ones':
            # self.feat_mask = nn.Parameter(torch.ones(1,7, 7, device=device))
            self.feat_mask = nn.Parameter(torch.ones(1, width, height, device=device) * 0.0001)  # 0.0001

        self.feature_is_continuous = feature_is_continuous
        self.num_levels = num_levels

        self.upsample = nn.UpsamplingBilinear2d(size=(width, height))

    def get_feat_mask(self, inv, apply_sigmoid=True, mask_temp=None):
        feat_mask = self.feat_mask
        feat_mask = 1 - feat_mask if inv else feat_mask
        return feat_mask

    def forward(self, bandwidths, sample_data, inv, mask_temp):
        x_batch, p_xj = sample_data
        _, num_channel, width, height = x_batch.shape
        bandwidths = bandwidths.reshape(1, num_channel, width, height)
        feat_mask = self.get_feat_mask(inv, mask_temp=mask_temp).detach()
        feat_mask = self.upsample(feat_mask.unsqueeze(0)).squeeze(0)

        if self.feature_is_continuous:
            diff = ((x_batch - self.target_image) * (1 / bandwidths)) ** 2
            selection_result = torch.exp(-0.5 * diff) * feat_mask.unsqueeze(0)
        else:
            # use Aitchison-Aitken kernel
            equal_mat = torch.eq(x_batch.to(device), self.target_image).float().to(device)
            equal_mat = equal_mat * feat_mask.unsqueeze(0)
            true_mat = (1 - self.bandwidths).unsqueeze(0).to(device)
            false_mat = (self.bandwidths / (self.num_levels - 1)).unsqueeze(0).to(device)
            selection_result = equal_mat * true_mat + (1 - equal_mat) * false_mat

        return selection_result


# X,M->Y
def get_perturb_prediction(model, sample_data, selection_results, y_type, perturb_func_name):
    sample, p_xj = sample_data
    assert sample.shape[0] == selection_results.shape[0]
    sample = perturb_func(sample, selection_results, perturb_func_name)

    predictions = model(sample)
    num_classes = predictions.shape[-1]

    if y_type == 'y_prob':
        predictions = predictions.softmax(dim=-1)
    elif y_type == 'y_pred':
        predictions = predictions.softmax(dim=-1)
        predictions = torch.argmax(predictions, dim=1)
    elif y_type == 'y_onehot':
        predictions = predictions.softmax(dim=-1)
        predictions = torch.argmax(predictions, dim=1)
        predictions = F.one_hot(predictions, num_classes=num_classes)  # (batch_size, num_classes)

    return predictions


class FANS(nn.Module):
    def __init__(self, x, dataset_name, feat_bandwidths, class_bandwidths, feature_is_continuous,
                 prediction_is_continuous, init_mask, perturb_func='mask'):
        super().__init__()
        dataset_name = dataset_name.lower()
        config_path = os.path.join(curr_dir, 'config.yaml')
        self.args = read_yaml(config_path)[dataset_name]
        _, num_channel, width, height = x.shape
        self.upsample = nn.UpsamplingBilinear2d(size=(width, height))

        if feature_is_continuous:
            self.psi = GaussianKernel()  # image data
        else:
            self.psi = AitchisonAitkenKernel(num_levels=2)  # graph data

        if prediction_is_continuous:
            self.phi = GaussianKernel()
        else:
            self.phi = AitchisonAitkenKernel(num_levels=2)

        self.target_x = x
        self.selector = FeatureSelector(self.target_x, self.args['init_method'], init_mask, feature_is_continuous=True)

        self.feature_is_continuous = feature_is_continuous
        self.prediction_is_continuous = prediction_is_continuous
        self.feat_bandwidths = nn.Parameter(feat_bandwidths)
        self.class_bandwidths = class_bandwidths

        self.target_class = None
        self.clean_onehot_target = None

        self.perturb_func_name = perturb_func

    def abduction(self, target_index, dataset, perturbed_y_onehots, feat_mask, clean_onehot_target, module_name):
        x_batch, p_xj = dataset
        perturbed_y_onehots = perturbed_y_onehots.to(device)
        inv = False if module_name == 'nec' else True
        feat_mask = self.upsample(feat_mask.unsqueeze(0)).squeeze(0)
        c_event_indictor = self.psi(self.feat_bandwidths.detach(), x_batch, self.target_x,
                                    feat_mask).squeeze()  # (n_sampels,)

        e_event_indictor = self.phi(self.class_bandwidths, perturbed_y_onehots, clean_onehot_target,
                                    inv=inv).squeeze()  # (n_samples,)

        if not self.feature_is_continuous:
            c_event_indictor = minmax_norm(c_event_indictor)

        if not self.prediction_is_continuous:
            e_event_indictor = minmax_norm(e_event_indictor)

        logK = torch.log(torch.sum(c_event_indictor))
        logN = torch.log(torch.tensor(len(c_event_indictor)))
        logV = torch.sum(torch.log(self.feat_bandwidths))

        log_pred_weight = (logK - logN - logV) / self.args['norm']

        assert not torch.isnan(log_pred_weight).any()
        assert not torch.isinf(log_pred_weight).any()

        pred_weight = torch.sigmoid(log_pred_weight)

        if module_name == 'nec':
            prob_ce_xj = c_event_indictor * e_event_indictor  # P(CE|x_j) (n_samples,)            
            # n_CE, (num_data, 1)
            ce_event_count = torch.sum(c_event_indictor * e_event_indictor)

            prob_ce = torch.sigmoid(ce_event_count * 100.)

            sample_weights_nec = prob_ce_xj * p_xj

            samples = resampling(x_batch, sample_weights_nec, resample_size=self.args['resample_size'])

            prob_ce = (prob_ce if torch.sum(prob_ce) != 0 else 1.) * self.args["coef_scale_nec"]

            return samples, prob_ce, pred_weight

        elif module_name == 'suf':
            e_event_indictor[0] = 1
            prob_invce_xj = c_event_indictor * e_event_indictor  # P(invCE|x_j) (n_samples,)

            # n_inv(CE), (num_data, 1)
            invce_event_count = torch.sum(c_event_indictor * e_event_indictor)
            prob_invce = invce_event_count / len(dataset)

            sample_weights_suf = prob_invce_xj * p_xj
            samples = resampling(x_batch, sample_weights_suf, resample_size=self.args['resample_size'])

            prob_invce = prob_invce * self.args["coef_scale_suf"]

            return samples, prob_invce, pred_weight

        else:
            assert False, f"Unknown module name: {module_name}"

    def action(self, edge_mask, module_name, temperature):
        if module_name == 'nec':
            edge_masks = []
            for _ in range(self.args['perturb_size']):
                multi_hot = concrete_sample(edge_mask.unsqueeze(0), temperature=1., bias=0., scale=self.args['scale'])[
                    0]  # bias=0.

                edge_masks.append(1 - multi_hot)

        elif module_name == 'suf':
            edge_masks = [torch.sigmoid(edge_mask)]

        else:
            assert False, f"Unknown module name: {module_name}"

        return edge_masks

    def prediction(self, dataset, feat_masks_perturbed, target_class, model, module_name, prediction_weight=None):
        probability = 0.
        loss = 0.
        clean_logit = model(self.target_x)[0]
        feat_masks_perturbed = torch.stack(feat_masks_perturbed).to(device)

        feat_masks_perturbed = self.upsample(feat_masks_perturbed)
        for xj in dataset:
            xj = xj.unsqueeze(0)
            xj = perturb_func(xj, feat_masks_perturbed, self.perturb_func_name)
            logits = model(xj)
            if torch.isnan(logits).any():
                print(xj)
                print(logits)
                raise

            if module_name == 'nec':
                diff_size = torch.norm(logits - clean_logit, p=2, dim=1)
                diff_size = torch.exp(-diff_size / 100)

                loss_i = torch.mean(diff_size)

                threshold = 1e-4
                is_close_to_zero = diff_size < threshold
                num_success = is_close_to_zero.count_nonzero().item()

            elif module_name == 'suf':
                one_hot_label = torch.eye(len(logits[0]))[target_class].unsqueeze(0).to(device)

                i, _ = torch.max((1 - one_hot_label) * logits, dim=1)
                j = torch.masked_select(logits, one_hot_label.bool())

                raw_loss = torch.clamp(i - j, min=0)  # (1,)

                num_success = raw_loss.numel() - raw_loss.count_nonzero()

                raw_loss = 1 - torch.exp(-raw_loss)

                loss_i = raw_loss.sum()

            else:
                assert False, f"Unknown module name: {module_name}"

            loss = loss + loss_i
            causal_effect = num_success / len(feat_masks_perturbed)
            probability = probability + causal_effect

        probability = probability / len(dataset)

        loss = loss * prediction_weight

        return probability, loss

    def forward(self, target_index, sample_data, model, temp, mask_temp=1.):
        feat_mask_logit = self.selector.get_feat_mask(inv=False, apply_sigmoid=False)
        feat_mask = self.selector.get_feat_mask(inv=False, apply_sigmoid=True, mask_temp=mask_temp)

        if self.clean_onehot_target is None:
            logit = model(self.target_x)[0]
            self.target_class = logit.argmax()
            self.clean_onehot_target = F.one_hot(self.target_class, num_classes=len(logit)).unsqueeze(
                0)  # (1, num_classes)

        # Nec module
        # Factual step
        selction_results = self.selector(self.feat_bandwidths.detach(), sample_data, inv=False, mask_temp=mask_temp)
        perturbed_ys_onehot = get_perturb_prediction(model, sample_data, selction_results, y_type='y_onehot',
                                                     perturb_func_name=self.perturb_func_name)

        # Counterfactual step
        dataset_nec, p_ce, pred_weight_nec = self.abduction(target_index, sample_data, perturbed_ys_onehot, feat_mask,
                                                            self.clean_onehot_target, module_name='nec')

        feat_masks_perturbed = self.action(feat_mask_logit, module_name='nec', temperature=temp)

        probability_nec, loss_nec = self.prediction(dataset_nec, feat_masks_perturbed, self.target_class, model,
                                                    module_name='nec', prediction_weight=pred_weight_nec)

        # Suf module
        inv_feat_mask = (1 - feat_mask).detach()
        # Factual step
        selction_results = self.selector(self.feat_bandwidths.detach(), sample_data, inv=True, mask_temp=mask_temp)
        perturbed_ys_onehot = get_perturb_prediction(model, sample_data, selction_results, y_type='y_onehot',
                                                     perturb_func_name=self.perturb_func_name)

        # Counterfactual step
        dataset_suf, p_ice, pred_weight_suf = self.abduction(target_index, sample_data, perturbed_ys_onehot,
                                                             inv_feat_mask, self.clean_onehot_target, module_name='suf')

        edge_masks_perturbed = self.action(feat_mask_logit, module_name='suf', temperature=temp)
        probability_suf, loss_suf = self.prediction(dataset_suf, edge_masks_perturbed, self.target_class, model,
                                                    module_name='suf', prediction_weight=pred_weight_suf)

        return probability_nec, probability_suf, p_ce * loss_nec, p_ice * loss_suf

    @property
    def explanation(self):
        mask_temp = get_temperature(self.args['epoch_pns'] - 1, self.args['temp_params'], self.args['epoch_pns'])
        feat_mask = self.selector.get_feat_mask(inv=False, apply_sigmoid=True, mask_temp=mask_temp)
        feat_mask = self.upsample(feat_mask.unsqueeze(0)).squeeze(0)
        return feat_mask.detach().cpu().numpy()


def explain(model, inputs, targets, **kwargs):
    model.eval()
    x_batch = inputs
    if isinstance(x_batch, np.ndarray):
        x_batch = torch.from_numpy(x_batch).to(device)

    config_path = os.path.join(curr_dir, 'config.yaml')
    dataset_name = kwargs.get('dataset_name', None)
    init_masks = kwargs.get('init_masks', None)
    # init_masks = None
    if init_masks is not None:
        init_masks = torch.from_numpy(init_masks).to(device)

    args = read_yaml(config_path)[dataset_name]

    class_bandwidths = compute_class_bandwidths(x_batch, model)

    explanations = []
    for i, x in enumerate(tqdm(x_batch, desc='Optimize Batch of Explantion', disable=False)):
        init_mask = None if init_masks is None else init_masks[i]
        # if len(explanations) > 6:
        #     break
        x = x.unsqueeze(0)  # (1, num_channel, width, height)
        _, num_channel, width, height = x.shape

        sample_data, feature_bandwidths = build_sample_data(x, size=10, )

        explainer = FANS(x, dataset_name, feature_bandwidths, class_bandwidths,
                         feature_is_continuous=True, prediction_is_continuous=False,
                         init_mask=init_mask, perturb_func=args['perturb_func']).to(device)

        optimizer = torch.optim.Adam(explainer.parameters(), lr=args['lr_pns'], amsgrad=True,
                                     weight_decay=args['weight_decay'])

        for epoch in tqdm(range(args['epoch_pns']), desc='One Explantion Optimizing', disable=True):
            explainer.train()
            optimizer.zero_grad()

            temp = get_temperature(epoch, args['temp_params'], args['epoch_pns'])

            nec_prob, suf_prob, nec_loss, suf_loss = explainer(i, sample_data, model, temp=1., mask_temp=temp)
            loss = nec_loss + suf_loss

            loss.backward()
            optimizer.step()

        explainer.eval()
        explanations.append(np.sum(explainer.explanation, axis=0, keepdims=True))

    explanations = np.array(explanations)

    return explanations
