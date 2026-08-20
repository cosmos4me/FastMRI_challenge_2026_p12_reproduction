import json
import zlib
import numpy as np
import torch

from utils.common.metrics import foreground_mask


def to_tensor(data):
    """Convert a NumPy array to a torch tensor."""
    return torch.from_numpy(data)


class DataTransform:
    def __init__(
        self,
        isforward,
        max_key,
        make_foreground=False,
        max_boxes=8,
        mri_augment=False,
        augment_seed=430,
        augment_start_epoch=5,
        augment_ramp_epochs=5,
        augment_max_probability=0.5,
        augment_max_shift=4,
        augment_coil_phase=True,
    ):
        self.isforward = isforward
        self.max_key = max_key
        self.make_foreground = make_foreground
        self.max_boxes = max_boxes
        self.mri_augment = bool(mri_augment and not isforward)
        self.augment_seed = int(augment_seed)
        self.augment_start_epoch = int(augment_start_epoch)
        self.augment_ramp_epochs = int(augment_ramp_epochs)
        self.augment_max_probability = float(augment_max_probability)
        self.augment_max_shift = int(augment_max_shift)
        self.augment_coil_phase = bool(augment_coil_phase)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def augmentation_probability(self):
        if not self.mri_augment:
            return 0.0
        current_epoch = self.epoch + 1
        if current_epoch <= self.augment_start_epoch:
            return 0.0
        if self.augment_ramp_epochs <= 0:
            return self.augment_max_probability
        progress = min(
            1.0,
            (current_epoch - self.augment_start_epoch)
            / self.augment_ramp_epochs,
        )
        return self.augment_max_probability * progress

    def _rng(self, fname, slice_index):
        filename_seed = zlib.crc32(fname.encode("utf-8")) & 0xFFFFFFFF
        sequence = np.random.SeedSequence(
            [self.augment_seed, self.epoch, filename_seed, int(slice_index)]
        )
        return np.random.default_rng(sequence)

    @staticmethod
    def _translate_kspace(kspace, shift_y, shift_x):
        """Apply an exact integer image translation to centered full k-space."""
        height, width = kspace.shape[-2:]
        frequency_y = np.fft.fftshift(np.fft.fftfreq(height))
        frequency_x = np.fft.fftshift(np.fft.fftfreq(width))
        phase = np.exp(
            -2j
            * np.pi
            * (
                frequency_y[:, None] * shift_y
                + frequency_x[None, :] * shift_x
            )
        )
        return np.asarray(kspace * phase, dtype=kspace.dtype)

    @staticmethod
    def _translate_boxes(boxes, shift_y, shift_x):
        translated = []
        for box in boxes:
            shifted = dict(box)
            shifted["x"] = int(box["x"]) + shift_x
            shifted["y"] = int(box["y"]) + shift_y
            translated.append(shifted)
        return translated

    def _apply_mri_augmentation(
        self, kspace, target, boxes, fname, slice_index
    ):
        probability = self.augmentation_probability()
        if probability <= 0.0:
            return kspace, target, boxes
        rng = self._rng(fname, slice_index)
        if rng.random() >= probability:
            return kspace, target, boxes

        shift_y = int(
            rng.integers(-self.augment_max_shift, self.augment_max_shift + 1)
        )
        shift_x = int(
            rng.integers(-self.augment_max_shift, self.augment_max_shift + 1)
        )
        if shift_y or shift_x:
            kspace = self._translate_kspace(kspace, shift_y, shift_x)
            target = np.roll(target, (shift_y, shift_x), axis=(-2, -1))
            boxes = self._translate_boxes(boxes, shift_y, shift_x)

        if self.augment_coil_phase:
            coil_phase = rng.uniform(-np.pi, np.pi, size=kspace.shape[0])
            shape = (kspace.shape[0],) + (1,) * (kspace.ndim - 1)
            phase = np.exp(1j * coil_phase).reshape(shape)
            kspace = np.asarray(kspace * phase, dtype=kspace.dtype)

        return (
            np.ascontiguousarray(kspace),
            np.ascontiguousarray(target),
            boxes,
        )

    def __call__(
        self, mask, input, target, attrs, fname, slice, acceleration=None
    ):
        input_data = input
        if not self.isforward:
            maximum = attrs[self.max_key]
            annotations = attrs.get("annotations", "{}")
            if isinstance(annotations, bytes):
                annotations = annotations.decode("utf-8")
            if isinstance(annotations, str):
                annotations = json.loads(annotations)
            slice_boxes = [
                dict(box) for box in annotations.get(str(slice), [])
            ]
            input_data, target, slice_boxes = self._apply_mri_augmentation(
                input_data,
                np.asarray(target),
                slice_boxes,
                fname,
                slice,
            )

            target = to_tensor(np.ascontiguousarray(target))
            bbox_mask = torch.zeros_like(target, dtype=torch.float32)
            foreground = (
                torch.from_numpy(foreground_mask(target.numpy())).float()
                if self.make_foreground
                else torch.zeros_like(target, dtype=torch.float32)
            )
            boxes = torch.zeros((self.max_boxes, 4), dtype=torch.int64)
            valid_boxes = []
            height, width = target.shape[-2:]
            for box in slice_boxes:
                x0 = max(0, int(box["x"]))
                y0 = max(0, int(box["y"]))
                x1 = min(width, int(box["x"]) + int(box["width"]))
                y1 = min(height, int(box["y"]) + int(box["height"]))
                if x1 - x0 >= 7 and y1 - y0 >= 7:
                    bbox_mask[y0:y1, x0:x1] = 1.0
                    valid_boxes.append((x0, y0, x1, y1))
            if len(valid_boxes) > self.max_boxes:
                raise ValueError(
                    f"{fname} slice {slice} has {len(valid_boxes)} boxes; "
                    f"increase --max-boxes above {self.max_boxes}"
                )
            if valid_boxes:
                boxes[:len(valid_boxes)] = torch.tensor(valid_boxes)
            if acceleration is None:
                acceleration = 4 if "acc4" in fname else 8
            bbox_data = {
                "mask": bbox_mask,
                "boxes": boxes,
                "count": torch.tensor(len(valid_boxes), dtype=torch.int64),
                "acceleration": torch.tensor(acceleration, dtype=torch.int64),
            }
        else:
            target = -1
            maximum = -1
            bbox_data = -1
            foreground = -1

        masked_input = np.ascontiguousarray(input_data * mask)
        kspace = to_tensor(masked_input)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(
            mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)
        ).byte()
        return (
            mask,
            kspace,
            target,
            maximum,
            bbox_data,
            foreground,
            fname,
            slice,
        )
