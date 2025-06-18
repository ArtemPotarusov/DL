import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.distributions.categorical import Categorical


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super().__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size, dataset.pad_id)
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        embeddings = self.embedding(indices)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_embeddings)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        logits = self.linear(output)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        # generated = prefix + ', а потом купил мужик шляпу, а она ему как раз.'
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        device = next(self.parameters()).device
        indices = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        indices = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
        hidden = None
        while indices.shape[1] < self.max_length:
            embeddings = self.embedding(indices)
            output, hidden = self.rnn(embeddings, hidden)
            logits = self.linear(output) / temp
            tokens = Categorical(logits=logits[:, -1:]).sample()
            indices = torch.cat([indices, tokens], dim=1)
            if tokens == self.dataset.eos_id:
                break
        return self.dataset.ids2text(indices.squeeze())
