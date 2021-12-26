from typing import Optional

from einops import repeat
from torch import nn
import torch
import math
import albumentations.augmentations.transforms as AF


class InputAdapter(nn.Module):
    def __init__(self, num_input_channels):
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels

    def forward(self, x):
        raise NotImplementedError()


class OutputAdapter(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self._output_shape = output_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        raise NotImplementedError()


class TextInputAdapter(InputAdapter):
    def __init__(self, vocab_size: int, max_seq_len: int, num_input_channels: int):
        super().__init__(num_input_channels=num_input_channels)

        self.text_embedding = nn.Embedding(vocab_size, num_input_channels)
        self.pos_encoding = nn.Parameter(torch.empty(max_seq_len, num_input_channels))

        self.scale = math.sqrt(num_input_channels)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.text_embedding.weight.data.uniform_(-0.1, 0.1)
            self.pos_encoding.uniform_(-0.5, 0.5)

    def forward(self, x):
        b, l = x.shape

        # repeat position encodings along batch dimension
        p_enc = repeat(self.pos_encoding[:l], '... -> b ...', b=b)

        return self.text_embedding(x) * self.scale + p_enc


class ClassificationOutputAdapter(OutputAdapter):
    def __init__(self,
                 num_classes: int,
                 num_outputs: int = 1,
                 num_output_channels: Optional[int] = None):

        if num_output_channels is None:
            num_output_channels = num_classes

        super().__init__(output_shape=(num_outputs, num_output_channels))
        self.linear = nn.Linear(num_output_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class TextOutputAdapter(ClassificationOutputAdapter):
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 num_output_channels: Optional[int] = None):
        super().__init__(num_classes=vocab_size,
                         num_outputs=max_seq_len,
                         num_output_channels=num_output_channels)


class TextMasking(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 unk_token_id: int = 1,
                 mask_token_id: int = 2,
                 num_special_tokens: int = 3,
                 mask_p: float = 0.15):
        """
        Text masking as described in https://arxiv.org/abs/1810.04805.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id
        self.num_special_tokens = num_special_tokens
        self.mask_p = mask_p
        self.masker = AF.GridDropout(p=1,holes_number_y=1,holes_number_x=1, ratio=0.5, random_offset=True)

    def forward(self, x, pad_mask):
        labels = x.clone()

        random_mask = torch.flatten(torch.tensor(self.masker(image=torch.ones((16, 16)).numpy())['image'])).bool().unsqueeze(0).repeat_interleave(x.shape[0], dim=0)

        # Mask special tokens in input (UNK, PAD)
        is_special = x == self.unk_token_id
        is_special |= pad_mask

        # Mask non-special tokens
        is_input = ~is_special

        # Randomly select 15% of non-special tokens
        is_selected = torch.rand_like(x, dtype=torch.float) < self.mask_p
        is_selected &= is_input

        # Of those, set 80% to MASK token, 10% to random token and leave 10% unchanged
        is_selected_1 = is_selected & (torch.rand_like(x, dtype=torch.float) < 0.9)
        is_selected_2 = is_selected_1 & (torch.rand_like(x, dtype=torch.float) < 1 / 9)
        x[is_selected_1] = self.mask_token_id
        # x[~random_mask] = self.mask_token_id
        #
        # labels[~random_mask] = self.mask_token_id
        # Based on the assumption that the id of the first
        # non-special token is self.num_special_tokens
        x[is_selected_2] = torch.randint(self.num_special_tokens,
                                         self.vocab_size,
                                         size=(is_selected_2.sum(),),
                                         device=x.device)

        # ignore labels of non-selected elements
        labels[~is_selected] = -100
        return x, labels