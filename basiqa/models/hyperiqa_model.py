import os
import pandas as pd
import torch,time
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, csv_write, tensor2img
from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x)
        # q = F.dropout(q)
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return q


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])



@MODEL_REGISTRY.register()
class HyperIQAModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(HyperIQAModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            if train_opt.get('pixel_opt2'):
              self.cri_pix2=build_loss(train_opt['pixel_opt2']).to(self.device)
            else :
              self.cri_pix2 = None
        else:
            self.cri_pix = None

        if self.cri_pix is None:
            raise ValueError('No loss found. Please use pix_loss in train setting.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        backbone_params = list(map(id, self.net_g.bbone.parameters()))
        self.hyper_net_params = filter(lambda p: id(p) not in backbone_params, self.net_g.parameters())
        paras = [{'params': self.hyper_net_params, 'lr': train_opt['optim_g']['lr']*0.1},
                 {'params': self.net_g.bbone.parameters(), 'lr': train_opt['optim_g']['lr']}
                 ]
        self.optimizer_g = torch.optim.Adam(paras, weight_decay=train_opt['optim_g']['weight_decay'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.image = data['image'].to(self.device)
        self.score = data['score'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        hyper_out=self.net_g(self.image)
        # Building target network
        model_target = TargetNet(hyper_out).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        # while 'paras['target_in_vec']' is the input to target net
        self.output = model_target(hyper_out['target_in_vec'])

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output.float(), self.score.float())
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_pix2:
            l_pix2 = self.cri_pix2(self.output.float(), self.score.float())
            l_total += l_pix2
            loss_dict['l_pix2'] = l_pix2

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                hyper_out = self.net_g_ema(self.image)
                # Building target network
                model_target = TargetNet(hyper_out).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                # while 'paras['target_in_vec']' is the input to target net
                self.output = model_target(hyper_out['target_in_vec'])
        else:
            self.net_g.eval()
            with torch.no_grad():
                hyper_out = self.net_g(self.image)
                # Building target network
                model_target = TargetNet(hyper_out).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                # while 'paras['target_in_vec']' is the input to target net
                self.output = model_target(hyper_out['target_in_vec'])
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        full_score = dataloader.dataset.opt.get('full_score', 1)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # method of metric was changed `cause iqa need to compute the srcc and plcc
        metric_data['pred'] = []
        metric_data['gt'] = []
        gt_all_finite = True
        name_list=[]  # for saving
        for idx, val_data in enumerate(dataloader):
            img_full_name=osp.basename(val_data['img_path'][0])
            img_name = osp.splitext(img_full_name)[0]
            name_list.append(img_full_name)  # for saving
            self.feed_data(val_data)
            self.test()
            metric_data['pred'].append(self.output.cpu().numpy())
            score_np = self.score.detach().cpu().numpy()
            if not np.isfinite(score_np).all():
                gt_all_finite = False
            metric_data['gt'].append(score_np)
            # tentative for out of GPU memory
            del self.score
            del self.image
            del self.output
            torch.cuda.empty_cache()
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        metric_data['pred']=np.array(metric_data['pred']).flatten()
        metric_data['gt']=np.array(metric_data['gt']).flatten()

        if not gt_all_finite:
            with_metrics = False

        if with_metrics:
            # calculate metrics
            for name, opt_ in self.opt['val']['metrics'].items():
                self.metric_results[name] += calculate_metric(metric_data, opt_)

        if save_img:
            if self.opt['is_train']:
                # save image is not supported in train state.
                pass
            else:
                if self.opt['val']['suffix']:
                    sav_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                        f'prediction_{self.opt["val"]["suffix"]}.csv')
                else:
                    sav_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                        f'prediction.csv')
                sav_csv = {
                    "file_name": name_list,
                    "prediction": metric_data['pred'],
                    "ground_truth": metric_data['gt']
                }
                sav_csv.update(self.metric_results)
                # print(sav_csv)
                sav_csv=pd.DataFrame(sav_csv)
                # sav=sav_csv.to_csv(sav_path)
                sav = csv_write(sav_csv, sav_path)

                try:
                    vis_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    pred_files = [
                        f for f in os.listdir(vis_dir)
                        if f.startswith('prediction') and f.endswith('.csv')
                    ]
                    pred_dfs = []
                    for f in pred_files:
                        df = pd.read_csv(osp.join(vis_dir, f))
                        cols = [c for c in ['file_name', 'prediction', 'ground_truth'] if c in df.columns]
                        if 'file_name' in cols and 'prediction' in cols:
                            pred_dfs.append(df[cols])
                    if len(pred_dfs) > 0:
                        all_pred = pd.concat(pred_dfs, ignore_index=True)
                        if 'ground_truth' not in all_pred.columns:
                            all_pred['ground_truth'] = np.nan

                        def _first_valid(series):
                            s = series.dropna()
                            return s.iloc[0] if len(s) > 0 else np.nan
                        results_df = all_pred.groupby('file_name', as_index=False).agg(
                            prediction=('prediction', 'mean'),
                            count=('prediction', 'size'),
                            ground_truth=('ground_truth', _first_valid)
                        )
                        results_df['prediction'] = results_df['prediction'] * full_score
                        if 'ground_truth' in results_df.columns:
                            results_df['ground_truth'] = results_df['ground_truth'] * full_score
                        results_path = osp.join(vis_dir, 'results.csv')
                        csv_write(results_df, results_path)
                        logger = get_root_logger()
                        logger.info(f'Please check averaged results: {results_path}')
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'Failed to generate averaged results.csv: {e}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                # self.metric_results[metric] /= (idx + 1)  # No need to do the divide for iqa metric
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        else:
            if not gt_all_finite:
                logger = get_root_logger()
                logger.info('Ground truth score is missing (NaN). Skip validation metrics.')

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        # print(self.metric_results.items())
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def save(self, epoch, current_iter,multi_round=0):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter,round=multi_round, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter,round=multi_round)
        self.save_training_state(epoch, current_iter)


    # rewrite the saving
    def save_network(self, net, net_label, current_iter,round, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{round}_{net_label}_{current_iter}.pth'
        save_path = osp.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

