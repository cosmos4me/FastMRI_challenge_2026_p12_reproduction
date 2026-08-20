import h5py
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.data.transforms import DataTransform

def _seed_worker(worker_id):
    """Seed NumPy/Python RNGs from the deterministic PyTorch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SliceData(Dataset):
    def __init__(
        self,
        root,
        transform,
        input_key,
        target_key,
        forward=False,
        cross_acc_remask=False,
        include_volume_context=False,
        all_anatomy_acc8=False,
        all_anatomy_acc4=False,
        acc8_mask_offset_augmentation=False,
        balanced_acc_offset_cycle=False,
        native_acceleration_filter=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.include_volume_context = bool(include_volume_context)
        self.cross_acc_remask = bool(cross_acc_remask and not forward)
        self.all_anatomy_acc8 = bool(all_anatomy_acc8 and not forward)
        self.all_anatomy_acc4 = bool(all_anatomy_acc4 and not forward)
        self.acc8_mask_offset_augmentation = bool(
            acc8_mask_offset_augmentation and not forward
        )
        self.balanced_acc_offset_cycle = bool(
            balanced_acc_offset_cycle and not forward
        )
        if self.all_anatomy_acc4 and self.all_anatomy_acc8:
            raise ValueError(
                "all-anatomy acc4 and acc8 remasking are mutually exclusive"
            )
        if self.acc8_mask_offset_augmentation and not self.all_anatomy_acc8:
            raise ValueError(
                "acc8 mask-offset augmentation requires all-anatomy acc8 remasking"
            )
        if self.balanced_acc_offset_cycle and (
            self.cross_acc_remask
            or self.all_anatomy_acc4
            or self.all_anatomy_acc8
            or self.acc8_mask_offset_augmentation
        ):
            raise ValueError(
                "balanced acc/offset cycle is mutually exclusive with legacy remasking"
            )
        self.all_anatomy_acceleration = (
            4 if self.all_anatomy_acc4
            else 8 if self.all_anatomy_acc8
            else None
        )
        self.native_acceleration_filter = native_acceleration_filter
        if self.native_acceleration_filter not in (None, 4, 8):
            raise ValueError("native acceleration filter must be 4, 8, or None")
        self.epoch = 0
        self.mask_offset_volume_base = 0
        self.image_examples = []
        self.kspace_examples = []
        self.bbox_positive = []
        self.forced_accelerations = []
        self.mask_bank = {}
        self.volume_indices = {}
        self.volume_stats = {}
        self.challenge_counts = {
            "total_slices": 0,
            "total_boxes": 0,
            "slice_counts": {4: 0, 8: 0},
            "box_counts": {4: 0, 8: 0},
        }

        if not forward:
            image_files = sorted((self.root / "image").iterdir())
            for fname in image_files:
                if (
                    self.native_acceleration_filter is not None
                    and self._filename_acceleration(fname.name)
                    != self.native_acceleration_filter
                ):
                    continue
                num_slices = self._get_metadata(fname)
                self.image_examples.extend(
                    (fname, slice_index) for slice_index in range(num_slices)
                )
                with h5py.File(fname, "r") as hf:
                    annotations = self._parse_annotations(
                        hf.attrs.get("annotations", "{}")
                    )
                    height, width = hf[self.target_key].shape[-2:]

                acceleration = self._filename_acceleration(fname.name)
                valid_per_slice = []
                for slice_index in range(num_slices):
                    valid = sum(
                        self._valid_box(box, height, width)
                        for box in annotations.get(str(slice_index), [])
                    )
                    valid_per_slice.append(valid)
                box_total = sum(valid_per_slice)
                self.challenge_counts["total_slices"] += num_slices
                self.challenge_counts["total_boxes"] += box_total
                self.challenge_counts["slice_counts"][acceleration] += num_slices
                self.challenge_counts["box_counts"][acceleration] += box_total
                self.bbox_positive.extend(count > 0 for count in valid_per_slice)
                self.volume_stats[fname.name] = {
                    "native_acceleration": acceleration,
                    "num_slices": num_slices,
                    "box_count": box_total,
                }

        kspace_files = sorted((self.root / "kspace").iterdir())
        for volume_index, fname in enumerate(kspace_files):
            if (
                self.native_acceleration_filter is not None
                and self._filename_acceleration(fname.name)
                != self.native_acceleration_filter
            ):
                continue
            num_slices = self._get_metadata(fname)
            self.volume_indices[fname.name] = volume_index
            if not self.forward:
                self.volume_stats[fname.name]["volume_index"] = volume_index
            self.kspace_examples.extend(
                (fname, slice_index) for slice_index in range(num_slices)
            )
            if (
                self.cross_acc_remask
                or self.all_anatomy_acceleration is not None
                or self.balanced_acc_offset_cycle
            ):
                with h5py.File(fname, "r") as hf:
                    native_mask = np.asarray(hf["mask"]).squeeze()
                acceleration = self._filename_acceleration(fname.name)
                if self.all_anatomy_acceleration is not None:
                    expected = self._fixed_equispaced_mask(
                        native_mask.size, acceleration, native_mask.dtype
                    )
                    if not np.array_equal(native_mask, expected):
                        raise ValueError(
                            "PromptMR specialists require the official mask pattern; "
                            f"unexpected mask in {fname.name}"
                        )
                key = (native_mask.size, acceleration)
                self.volume_stats[fname.name]["mask_size"] = native_mask.size
                previous = self.mask_bank.get(key)
                if previous is not None and not np.array_equal(
                    previous, native_mask
                ):
                    raise ValueError(
                        f"multiple masks found for width={native_mask.size}, "
                        f"acc={acceleration}; an explicit multi-mask bank is required"
                    )
                self.mask_bank[key] = native_mask.copy()

        if self.all_anatomy_acceleration is not None:
            widths = sorted({width for width, _ in self.mask_bank})
            for width in widths:
                key = (width, self.all_anatomy_acceleration)
                if key in self.mask_bank:
                    continue
                template = next(
                    mask
                    for (mask_width, _), mask in self.mask_bank.items()
                    if mask_width == width
                )
                self.mask_bank[key] = self._fixed_equispaced_mask(
                    width, self.all_anatomy_acceleration, template.dtype
                )

        elif self.balanced_acc_offset_cycle:
            widths = sorted({width for width, _ in self.mask_bank})
            for width in widths:
                template = next(
                    mask
                    for (mask_width, _), mask in self.mask_bank.items()
                    if mask_width == width
                )
                for acceleration in (4, 8):
                    key = (width, acceleration)
                    if key not in self.mask_bank:
                        self.mask_bank[key] = self._fixed_equispaced_mask(
                            width, acceleration, template.dtype
                        )

        if not forward and len(self.image_examples) != len(self.kspace_examples):
            raise ValueError("image and kspace slice counts do not match")

        if not forward:
            base_count = len(self.kspace_examples)
            if self.all_anatomy_acceleration is not None:
                self.forced_accelerations = [self.all_anatomy_acceleration] * base_count
                self.challenge_counts["slice_counts"] = {
                    4: self.challenge_counts["total_slices"] if self.all_anatomy_acc4 else 0,
                    8: self.challenge_counts["total_slices"] if self.all_anatomy_acc8 else 0,
                }
                self.challenge_counts["box_counts"] = {
                    4: self.challenge_counts["total_boxes"] if self.all_anatomy_acc4 else 0,
                    8: self.challenge_counts["total_boxes"] if self.all_anatomy_acc8 else 0,
                }
            else:
                self.forced_accelerations = [None] * base_count

    @staticmethod
    def _parse_annotations(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value) if isinstance(value, str) else value

    @staticmethod
    def _filename_acceleration(name):
        if "acc4" in name:
            return 4
        if "acc8" in name:
            return 8
        raise ValueError(f"cannot infer acceleration from training file: {name}")

    @staticmethod
    def _fixed_equispaced_mask(
        width, acceleration, dtype=np.float32, offset=None
    ):
        """Reproduce the fixed challenge mask for a missing width/acc pair.

        Every provided 2026 train mask follows this same 8%-ACS and centered
        equispaced-offset rule. Synthesizing only a missing width avoids
        discarding fully sampled anatomy solely because the native acc8 split
        did not contain that matrix width.
        """
        if acceleration not in (4, 8):
            raise ValueError("fixed challenge mask acceleration must be 4 or 8")
        width = int(width)
        acs_length = round(0.08 * width) - 1
        acs_start = (width - acs_length) // 2
        acs_end = acs_start + acs_length
        mask = np.zeros(width, dtype=dtype)
        mask[acs_start:acs_end + 1] = 1
        start_index = (
            (width // 2) % acceleration if offset is None else int(offset)
        )
        if start_index < 0 or start_index >= acceleration:
            raise ValueError(
                f"mask offset must be in [0, {acceleration - 1}]"
            )
        mask[start_index::acceleration] = 1
        return mask

    @staticmethod
    def _valid_box(box, height, width):
        x0 = max(0, int(box["x"]))
        y0 = max(0, int(box["y"]))
        x1 = min(width, int(box["x"]) + int(box["width"]))
        y1 = min(height, int(box["y"]) + int(box["height"]))
        return int(x1 - x0 >= 7 and y1 - y0 >= 7)

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf:
                return hf[self.input_key].shape[0]
            if self.target_key in hf:
                return hf[self.target_key].shape[0]
        raise KeyError(f"neither {self.input_key} nor {self.target_key} in {fname}")

    def set_epoch(self, epoch):
        """Select one acceleration per volume, flipping every epoch."""
        self.epoch = int(epoch)
        if hasattr(self.transform, "set_epoch"):
            self.transform.set_epoch(epoch)

    def augmentation_probability(self):
        if hasattr(self.transform, "augmentation_probability"):
            return self.transform.augmentation_probability()
        return 0.0

    def get_challenge_counts(self):
        if not self.cross_acc_remask and not self.balanced_acc_offset_cycle:
            return self.challenge_counts
        counts = {
            "total_slices": self.challenge_counts["total_slices"],
            "total_boxes": self.challenge_counts["total_boxes"],
            "slice_counts": {4: 0, 8: 0},
            "box_counts": {4: 0, 8: 0},
        }
        for stats in self.volume_stats.values():
            phase = stats["volume_index"]
            if self.balanced_acc_offset_cycle:
                phase += self.mask_offset_volume_base
            desired = 4 if (phase + self.epoch) % 2 == 0 else 8
            key = (stats["mask_size"], desired)
            acceleration = desired if key in self.mask_bank else stats["native_acceleration"]
            counts["slice_counts"][acceleration] += stats["num_slices"]
            counts["box_counts"][acceleration] += stats["box_count"]
        return counts

    def get_balanced_acc_offset_counts(self):
        """Return current volume/slice coverage over all 12 mask states."""
        counts = {
            acceleration: {
                offset: {"volumes": 0, "slices": 0}
                for offset in range(acceleration)
            }
            for acceleration in (4, 8)
        }
        if not self.balanced_acc_offset_cycle:
            return counts
        for stats in self.volume_stats.values():
            phase = self.mask_offset_volume_base + stats["volume_index"]
            acceleration = 4 if (phase + self.epoch) % 2 == 0 else 8
            offset = (self.epoch // 2 + phase) % acceleration
            counts[acceleration][offset]["volumes"] += 1
            counts[acceleration][offset]["slices"] += stats["num_slices"]
        return counts

    def get_acc8_mask_offset_counts(self):
        """Return per-epoch volume/slice coverage for P07 mask offsets."""
        counts = {
            offset: {"volumes": 0, "slices": 0}
            for offset in range(8)
        }
        if not self.acc8_mask_offset_augmentation:
            return counts
        for stats in self.volume_stats.values():
            offset = (
                self.mask_offset_volume_base
                + stats["volume_index"]
                + self.epoch
            ) % 8
            counts[offset]["volumes"] += 1
            counts[offset]["slices"] += stats["num_slices"]
        return counts

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, index):
        if not self.forward:
            image_fname, image_slice = self.image_examples[index]
        kspace_fname, data_slice = self.kspace_examples[index]
        if not self.forward:
            if image_fname.name != kspace_fname.name or image_slice != data_slice:
                raise ValueError(
                    f"image {image_fname.name}:{image_slice} does not match "
                    f"kspace {kspace_fname.name}:{data_slice}"
                )

        with h5py.File(kspace_fname, "r") as hf:
            input_data = hf[self.input_key][data_slice]
            mask = np.asarray(hf["mask"]).squeeze()
            adjacent_data = None
            if self.include_volume_context:
                num_slices = hf[self.input_key].shape[0]
                adjacent_indices = (
                    max(0, data_slice - 1),
                    min(num_slices - 1, data_slice + 1),
                )
                adjacent_data = np.stack(
                    [hf[self.input_key][item] for item in adjacent_indices]
                )

        acceleration = None
        forced_acceleration = (
            self.forced_accelerations[index] if not self.forward else None
        )
        if self.balanced_acc_offset_cycle:
            volume_index = self.volume_indices[kspace_fname.name]
            phase = self.mask_offset_volume_base + volume_index
            acceleration = 4 if (phase + self.epoch) % 2 == 0 else 8
            offset = (self.epoch // 2 + phase) % acceleration
            mask = self._fixed_equispaced_mask(
                mask.size, acceleration, mask.dtype, offset=offset
            )
        elif forced_acceleration is not None:
            if self.acc8_mask_offset_augmentation:
                volume_index = self.volume_indices[kspace_fname.name]
                offset = (
                    self.mask_offset_volume_base + volume_index + self.epoch
                ) % 8
                replacement = self._fixed_equispaced_mask(
                    mask.size, 8, mask.dtype, offset=offset
                )
            else:
                replacement = self.mask_bank.get((mask.size, forced_acceleration))
            if replacement is None:
                raise ValueError(
                    f"no acc{forced_acceleration} mask for width={mask.size}"
                )
            mask = replacement
            acceleration = forced_acceleration
        elif self.cross_acc_remask:
            native_acceleration = self._filename_acceleration(kspace_fname.name)
            volume_index = self.volume_indices[kspace_fname.name]
            desired_acceleration = 4 if (volume_index + self.epoch) % 2 == 0 else 8
            replacement = self.mask_bank.get((mask.size, desired_acceleration))
            if replacement is not None:
                mask = replacement
                acceleration = desired_acceleration
            else:
                acceleration = native_acceleration
        elif not self.forward:
            acceleration = self._filename_acceleration(kspace_fname.name)

        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][data_slice]
                attrs = dict(hf.attrs)

        sample = self.transform(
            mask,
            input_data,
            target,
            attrs,
            kspace_fname.name,
            data_slice,
            acceleration,
        )
        if not self.include_volume_context:
            return sample

        def complex_tensor(array):
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            return torch.stack((tensor.real, tensor.imag), dim=-1)

        masked_adjacent = np.ascontiguousarray(
            adjacent_data * mask
        )
        h16_payload = {
            "adjacent_kspace": complex_tensor(masked_adjacent),
            # Training uses this target only in an auxiliary loss. The fixed
            # inference path never forwards it to the model.
            "full_kspace": complex_tensor(input_data),
        }
        return sample + (h16_payload,)


class MultiRootSliceData(Dataset):
    """Present multiple independent SliceData roots as one training dataset."""

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("MultiRootSliceData requires at least one dataset")
        self.datasets = list(datasets)
        self.offsets = []
        total = 0
        volume_total = 0
        for dataset in self.datasets:
            dataset.mask_offset_volume_base = volume_total
            volume_total += len(dataset.volume_stats)
            total += len(dataset)
            self.offsets.append(total)
        self.bbox_positive = [
            value for dataset in self.datasets for value in dataset.bbox_positive
        ]
        self.challenge_counts = self.get_challenge_counts()

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        previous = 0
        for dataset, stop in zip(self.datasets, self.offsets):
            if index < stop:
                return dataset[index - previous]
            previous = stop
        raise IndexError(index)

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def augmentation_probability(self):
        return self.datasets[0].augmentation_probability()

    def get_challenge_counts(self):
        combined = {
            "total_slices": 0,
            "total_boxes": 0,
            "slice_counts": {4: 0, 8: 0},
            "box_counts": {4: 0, 8: 0},
        }
        for dataset in self.datasets:
            counts = dataset.get_challenge_counts()
            combined["total_slices"] += counts["total_slices"]
            combined["total_boxes"] += counts["total_boxes"]
            for acceleration in (4, 8):
                combined["slice_counts"][acceleration] += counts[
                    "slice_counts"
                ][acceleration]
                combined["box_counts"][acceleration] += counts[
                    "box_counts"
                ][acceleration]
        return combined

    def get_acc8_mask_offset_counts(self):
        combined = {
            offset: {"volumes": 0, "slices": 0}
            for offset in range(8)
        }
        for dataset in self.datasets:
            for offset, counts in dataset.get_acc8_mask_offset_counts().items():
                combined[offset]["volumes"] += counts["volumes"]
                combined[offset]["slices"] += counts["slices"]
        return combined


    def get_balanced_acc_offset_counts(self):
        combined = {
            acceleration: {
                offset: {"volumes": 0, "slices": 0}
                for offset in range(acceleration)
            }
            for acceleration in (4, 8)
        }
        for dataset in self.datasets:
            local = dataset.get_balanced_acc_offset_counts()
            for acceleration in (4, 8):
                for offset in range(acceleration):
                    combined[acceleration][offset]["volumes"] += local[
                        acceleration
                    ][offset]["volumes"]
                    combined[acceleration][offset]["slices"] += local[
                        acceleration
                    ][offset]["slices"]
        return combined


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if not isforward:
        max_key = args.max_key
        target_key = args.target_key
    else:
        max_key = -1
        target_key = -1

    model_type = getattr(args, "model_type", "")
    specialist_acceleration = {
        "p00_acc8_promptmr_plus": 8,
        "p00m_acc8_promptmr_plus": 8,
        "p00s_acc8_promptmr_plus": 8,
        "p01m_acc4_promptmr_plus": 4,
        "p02m_acc4_promptmr_plus": 4,
        "p07m_acc8_multimask_promptmr_plus": 8,
        "p11m_acc8_sampling_aware_promptmr_plus": 8,
    }.get(model_type)

    challenge_loss = getattr(args, "loss_mode", "legacy") == "challenge"
    transform = DataTransform(
            isforward,
            max_key,
            make_foreground=(
                challenge_loss
                or getattr(args, "foreground_loss_weight", 0.0) > 0
            ),
            max_boxes=getattr(args, "max_boxes", 8),
            mri_augment=(
                getattr(args, "mri_augment", False) and shuffle
            ),
            augment_seed=getattr(args, "seed", 430),
            augment_start_epoch=getattr(
                args, "mri_augment_start_epoch", 5
            ),
            augment_ramp_epochs=getattr(
                args, "mri_augment_ramp_epochs", 5
            ),
            augment_max_probability=getattr(
                args, "mri_augment_max_prob", 0.5
            ),
            augment_max_shift=getattr(
                args, "mri_augment_max_shift", 4
            ),
            augment_coil_phase=getattr(
                args, "mri_augment_coil_phase", False
            ),
    )

    roots = [data_path]
    if shuffle and not isforward:
        roots.extend(getattr(args, "extra_data_path_train", []) or [])
    elif not isforward:
        roots.extend(getattr(args, "extra_data_path_val", []) or [])
    datasets = [
        SliceData(
            root=root,
            transform=transform,
            input_key=args.input_key,
            target_key=target_key,
            forward=isforward,
            cross_acc_remask=(
                getattr(args, "cross_acc_remask", False) and shuffle
            ),
            include_volume_context=(
                getattr(args, "model_type", "") == "h16_adjacent_kspace_varnet"
            ),
            all_anatomy_acc8=(
                getattr(args, "all_anatomy_acc8", False) and shuffle
            ),
            all_anatomy_acc4=(
                getattr(args, "all_anatomy_acc4", False) and shuffle
            ),
            acc8_mask_offset_augmentation=(
                getattr(args, "acc8_mask_offset_augmentation", False)
                and shuffle
            ),
            balanced_acc_offset_cycle=(
                getattr(args, "balanced_acc_offset_cycle", False)
                and shuffle
            ),
            native_acceleration_filter=(
                specialist_acceleration
                if (
                    specialist_acceleration is not None
                    and not shuffle
                    and not isforward
                )
                else None
            ),
        )
        for root in roots
    ]
    data_storage = (
        datasets[0] if len(datasets) == 1 else MultiRootSliceData(datasets)
    )

    sampler = None
    bbox_sample_weight = getattr(args, "bbox_sample_weight", 1.0)
    if not isforward and shuffle and bbox_sample_weight > 1.0:
        weights = [
            bbox_sample_weight if positive else 1.0
            for positive in data_storage.bbox_positive
        ]
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )

    num_workers = int(getattr(args, "num_workers", 0))
    loader_kwargs = dict(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(getattr(args, "pin_memory", False)),
    )
    if num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=int(getattr(args, "prefetch_factor", 2)),
            worker_init_fn=_seed_worker,
            # SliceData.set_epoch() changes remasking/augmentation every epoch.
            # Recreating workers propagates that state into each worker copy.
            persistent_workers=False,
        )
    return DataLoader(**loader_kwargs)
