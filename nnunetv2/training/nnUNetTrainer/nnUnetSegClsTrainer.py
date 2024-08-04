from typing import Union, Type, List, Tuple
import numpy as np
import torch
import pydoc
from time import time, sleep

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn, autocast
from torch import distributed as dist

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from nnunetv2.training.dataloading.data_loader_segcls_2d import nnUNetSegClsDataLoader2D
from nnunetv2.training.dataloading.data_loader_segcls_3d import nnUNetSegClsDataLoader3D
from nnunetv2.training.dataloading.segcls_dataset import nnUNetSegClsDataset

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.logging.nnunet_logger import nnUnetSegClsLogger

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA



class nnUnetSegClsTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.initial_lr = 1e-2
        self.alpha = 0.5
        self.logger = nnUnetSegClsLogger()

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        ## edit this part to change the network architecture
        architecture_kwargs = dict(**arch_init_kwargs)
        print(architecture_kwargs)

        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])


        if enable_deep_supervision is not None and 'deep_supervision' not in arch_init_kwargs.keys():
            architecture_kwargs['deep_supervision'] = enable_deep_supervision

        # Extracting arguments that are specifically passed to the constructor to avoid duplication
        n_stages = architecture_kwargs.pop('n_stages')
        features_per_stage = architecture_kwargs.pop('features_per_stage')
        conv_op = architecture_kwargs.pop('conv_op')
        kernel_sizes = architecture_kwargs.pop('kernel_sizes')
        strides = architecture_kwargs.pop('strides')
        n_conv_per_stage = architecture_kwargs.pop('n_conv_per_stage')
        n_conv_per_stage_decoder = architecture_kwargs.pop('n_conv_per_stage_decoder')
        conv_bias = architecture_kwargs.pop('conv_bias')
        norm_op = architecture_kwargs.pop('norm_op')
        norm_op_kwargs = architecture_kwargs.pop('norm_op_kwargs')
        dropout_op = architecture_kwargs.pop('dropout_op')
        dropout_op_kwargs = architecture_kwargs.pop('dropout_op_kwargs')
        nonlin = architecture_kwargs.pop('nonlin')
        nonlin_kwargs = architecture_kwargs.pop('nonlin_kwargs')

        network = SegClsNet(num_input_channels,
                            n_stages=n_stages,
                            features_per_stage=features_per_stage,
                            conv_op=conv_op,
                            kernel_sizes=kernel_sizes,
                            strides=strides,
                            n_conv_per_stage=n_conv_per_stage,
                            num_classes=num_output_channels,
                            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                            conv_bias=conv_bias,
                            norm_op=norm_op,
                            norm_op_kwargs=norm_op_kwargs,
                            dropout_op=dropout_op,
                            dropout_op_kwargs=dropout_op_kwargs,
                            nonlin=nonlin,
                            nonlin_kwargs=nonlin_kwargs,
                            **architecture_kwargs)

        return network

    def _build_loss(self):
        if self.label_manager.has_regions:
            segmentation_loss = DC_and_BCE_loss({},
                                {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                use_ignore_label=self.label_manager.ignore_label is not None,
                                dice_class=MemoryEfficientSoftDiceLoss)
        else:
            segmentation_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                            ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        classification_loss = nn.CrossEntropyLoss()

        if self._do_i_compile():
            segmentation_loss.dc = torch.compile(segmentation_loss.dc)
            classification_loss = torch.compile(classification_loss) # for consistency

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            segmentation_loss = DeepSupervisionWrapper(segmentation_loss, weights)

        return segmentation_loss, classification_loss

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetSegClsDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0,
                                   load_classification_labels=True)
        dataset_val = nnUNetSegClsDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0,
                                    load_classification_labels=True)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetSegClsDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetSegClsDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        else:
            dl_tr = nnUNetSegClsDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetSegClsDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        seg_target = batch['target']
        cls_target = batch['cls_target']

        data = data.to(self.device, non_blocking=True)

        if isinstance(seg_target, list):
            seg_target = [st.to(self.device, non_blocking=True) for st in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)
        
        if isinstance(cls_target, list):
            cls_target = [ct.to(self.device, non_blocking=True) for ct in cls_target]
        else:
            cls_target = cls_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_output = self.network(data)

            seg_loss = self.loss[0](seg_output, seg_target)
            cls_loss = self.loss[1](cls_output, cls_target)
            total_loss = seg_loss + (self.alpha * cls_loss)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'seg_loss': seg_loss.detach().cpu().numpy(), 'cls_loss': cls_loss.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        seg_target = batch['target']
        cls_target = batch['cls_target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(seg_target, list):
            seg_target = [st.to(self.device, non_blocking=True) for st in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)
        
        if isinstance(cls_target, list):
            cls_target = [ct.to(self.device, non_blocking=True) for ct in cls_target]
        else:
            cls_target = cls_target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_output = self.network(data)
            del data
            seg_loss = self.loss[0](seg_output, seg_target)
            cls_loss = self.loss[1](cls_output, cls_target)
        
            total_loss = seg_loss + (self.alpha * cls_loss)

        if self.enable_deep_supervision:
            seg_output = seg_output[0]
            target = seg_target[0] # not chaning target to seg_target everywhere. 

        axes = [0] + list(range(2, seg_output.ndim))


        output = seg_output

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'seg_loss': seg_loss.detach().cpu().numpy(), 
                'tp_hard': tp_hard, 
                'fp_hard': fp_hard, 
                'fn_hard': fn_hard, 
                'total_loss': total_loss.detach().cpu().numpy(), 
                'cls_loss': cls_loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            seg_losses_tr = [None for _ in range(dist.get_world_size())]
            cls_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(seg_losses_tr, outputs['seg_loss'])
            dist.all_gather_object(cls_losses_tr, outputs['cls_loss'])
            seg_loss_here = np.vstack(seg_losses_tr).mean()
            cls_loss_here = np.vstack(cls_losses_tr).mean()
        else:
            seg_loss_here = np.mean(outputs['seg_loss'])
            cls_loss_here = np.mean(outputs['cls_loss'])

        self.logger.log('train_seg_losses', seg_loss_here, self.current_epoch)
        self.logger.log('train_cls_losses', cls_loss_here, self.current_epoch)


    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            seg_losses_val = [None for _ in range(dist.get_world_size())]
            cls_losses_val = [None for _ in range(dist.get_world_size())]

            dist.all_gather_object(seg_losses_val, outputs_collated['seg_loss'])
            dist.all_gather_object(cls_losses_val, outputs_collated['cls_loss'])

            seg_loss_here = np.vstack(seg_losses_val).mean()
            cls_loss_here = np.vstack(cls_losses_val).mean()
        else:
            seg_loss_here = np.mean(outputs_collated['seg_loss'])
            cls_loss_here = np.mean(outputs_collated['cls_loss'])


        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_seg_losses', seg_loss_here, self.current_epoch)
        self.logger.log('val_cls_losses', cls_loss_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file('train_seg_loss', np.round(self.logger.my_fantastic_logging['train_seg_losses'][-1], decimals=4))
        self.print_to_log_file('train_cls_loss', np.round(self.logger.my_fantastic_logging['train_cls_losses'][-1], decimals=4))
        self.print_to_log_file('val_seg_loss', np.round(self.logger.my_fantastic_logging['val_seg_losses'][-1], decimals=4))
        self.print_to_log_file('val_cls_loss', np.round(self.logger.my_fantastic_logging['val_cls_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                            self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


class SegClsNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        # print(f"features_per_stage: {features_per_stage[-1]}")
        self.fc = nn.Linear(features_per_stage[-1], 3)

    def forward(self, x):
        skips = self.encoder(x)
        # print skips shape
        # print(skips[-1].shape) # torch.Size([3, 32, 64, 128, 192])

        # apply classifier head to the last layer of the encoder. For simplicity. 
        # Can have more complex head with attention applied to all layers.

        encoded_features = skips[-1] # skips[-1].shape = torch.Size([3, 320, 4, 4, 6])
        pooled_features = self.global_avg_pool(encoded_features).view(encoded_features.size(0), -1) # pooled_features: torch.Size([3, 320])

        # print(f"pooled_features: {pooled_features.shape}")

        classification_output = self.fc(pooled_features)

        return self.decoder(skips), classification_output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)