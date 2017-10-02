import numpy as np
import torch
import math
from torch.autograd import Variable


def _get_normalized_masked_log_probablities(vector, mask):
    # We calculate normalized log probabilities in a numerically stable fashion, as done
    # in https://github.com/rkadlec/asreader/blob/master/asreader/custombricks/softmax_mask_bricks.py
    input_masked = mask * vector
    shifted = mask * (input_masked - input_masked.max(dim=1, keepdim=True)[0])
    # We add epsilon to avoid numerical instability when the sum in the log yields 0.
    normalization_constant = ((mask * shifted.exp()).sum(dim=1, keepdim=True) + 1e-7).log()
    normalized_log_probabilities = (shifted - normalization_constant)
    return normalized_log_probabilities


def masked_softmax(vector, mask):

    if mask is not None:
        return mask * _get_normalized_masked_log_probablities(vector, mask).exp()
    else:
        # There is no mask, so we use the provided ``torch.nn.functional.softmax`` function.
        return torch.nn.functional.softmax(vector)


def last_dim_softmax(tensor, mask):

    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor.size()[-1])
    if mask is not None:
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(tensor).contiguous().float()
        mask = mask.view(-1, mask.size()[-1])
    reshaped_result = masked_softmax(reshaped_tensor, mask)
    return reshaped_result.view(*tensor_shape)


def weighted_sum(matrix, attention):

    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def replace_masked_values(tensor, mask, replace_with):

    # We'll build a tensor of the same shape as `tensor`, zero out masked values, then add back in
    # the `replace_with` value.
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))

    one_minus_mask = 1 - mask
    values_to_add = replace_with * one_minus_mask
    return tensor * mask + values_to_add


class DotProductSimilarity(torch.nn.Module):

    def __init__(self, scale_output: bool = False) -> None:
        super(DotProductSimilarity, self).__init__()
        self._scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result


class MatrixAttention(torch.nn.Module):
    def __init__(self):
        super(MatrixAttention, self).__init__()
        self._similarity_function = DotProductSimilarity()

    def forward(self, matrix_1, matrix_2):
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])

        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)


class BiAttentionLayer(torch.nn.Module):
    def __init__(self):
        super(BiAttentionLayer, self).__init__()
        self.matrix_attention = MatrixAttention()

    def forward(self, encoded_passage, encoded_question, passage_mask, question_mask):

        passage_mask = passage_mask.type(torch.FloatTensor)
        question_mask = question_mask.type(torch.FloatTensor)

        batch_size = encoded_passage.size()[0]
        passage_length = encoded_passage.size()[1]
        encoding_dim = encoded_passage.size()[2]

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self.matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = replace_masked_values(passage_question_similarity,
                                               question_mask.unsqueeze(1),
                                               -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                             passage_length,
                                                                             encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                  passage_question_vectors,
                                  encoded_passage * passage_question_vectors,
                                  encoded_passage * tiled_question_passage_vector],
                                  dim=-1)

        return final_merged_passage