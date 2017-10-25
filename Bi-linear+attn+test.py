
# coding: utf-8

# # Reasonet evaluation

# Части, специфичные для Reasonet

# In[1]:


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger.info('All works')


# In[2]:


from overrides import overrides

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from allennlp.common import Params
#from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.modules.similarity_function import SimilarityFunction
from allennlp.nn import Activation
from torch.autograd import Variable
from allennlp.modules.similarity_functions.bilinear import BilinearSimilarity
from allennlp.modules.attention import Attention
from allennlp.nn.util import weighted_sum
import allennlp.nn.util as util


# In[3]:


def CUDA_wrapper(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


# In[4]:


def batchwise_index(array_batch, index_batch):
    assert index_batch.dim() == 1
    assert array_batch.size(0) == index_batch.size(0)
    index_batch_one_hot = CUDA_wrapper(
        autograd.Variable(torch.ByteTensor(array_batch.size()).zero_(), requires_grad=False)
    )
    index_batch_one_hot.scatter_(1, index_batch.data.unsqueeze(-1), 1)
    return array_batch[index_batch_one_hot]


# # Attention over memory

# In[5]:


class AttentionCoefProvider(object):
    """
    Attention coef provider
    """
    def __init__(self,
                 similarity_function: SimilarityFunction,
                 normalize: bool) -> None:
        self._similarity_function = similarity_function
        self._attention = Attention(similarity_function, normalize=normalize)
        super(AttentionCoefProvider, self).__init__()
        
    def forward(self, state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self._attention(state, memory)


# In[6]:


class AttentionOverMemory(object):
    """
    Attention over memory
    """
    def __init__(self,
                 coef_provider: AttentionCoefProvider) -> None:
        self._coef_provider = coef_provider
        super(AttentionOverMemory, self).__init__()
        
    def forward(self, state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn_coefficients = self._coef_provider.forward(state, memory)
        return weighted_sum(memory, attn_coefficients)


# In[7]:


AttentionProvider = AttentionOverMemory(AttentionCoefProvider(BilinearSimilarity(300,300), True))


# # State Controller

# In[8]:


class StateController(object):

    def __init__(self, module: torch.nn.modules.RNNCell) -> None:
        super(StateController, self).__init__()
        self._module = module
        self.hidden_state = None
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        return self._module.hidden_size

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:

        if hidden_state is None:
            if not self.hidden_state is None:
                hidden_state = self.hidden_state
            else:
                raise ConfigurationError("Hidden state must be specified!")
                
        self.hidden_state = self._module(inputs, hidden_state)
        return self.hidden_state


# In[9]:


state_controller = StateController(torch.nn.GRUCell(300, 300))


# ## Termination gate

# In[10]:


class TerminationGate(nn.Module):
    
    def __init__(self, hidden_dim: int) -> None:
        super(TerminationGate, self).__init__()
        self._hiden_dim = hidden_dim
        self.linear = torch.nn.Linear(hidden_dim, 2)
        
    def forward(self, hidden_state: torch.Tensor):
        tensor_output = util.last_dim_softmax(self.linear(hidden_state))
        last_dim = tensor_output.dim() - 1
        return torch.chunk(tensor_output, 2, last_dim)


# ## Reasonet inner controller logic

# In[11]:


class ReasoningProcess(object):
    
    def __init__(self, timesteps:int,
                 attention_provider: AttentionProvider,
                 state_controller: StateController) -> None:
        
        self._timesteps = timesteps
        self._attention_provider = attention_provider
        self._state_controller = state_controller
        
    def forward(self, initial_hidden_state: torch.Tensor,
                memory: torch.Tensor):
        
        # use initial_hidden state to perform first state of computations
        attn = self._attention_provider.forward(initial_hidden_state, memory)
        hidden_state = self._state_controller.forward(attn, initial_hidden_state)
        
        if self._timesteps > 1:
            for i in range(self._timesteps-1):
                attn = self._attention_provider.forward(hidden_state, memory)
                hidden_state = self._state_controller.forward(attn)
                
        return hidden_state
                


# In[12]:


class ReasoningProcessStep(object):
    
    def __init__(self,
                 attention_provider: AttentionProvider,
                 state_controller: StateController) -> None:
        
        self._attention_provider = attention_provider
        self._state_controller = state_controller
        
    def forward(self, initial_hidden_state: torch.Tensor,
                memory: torch.Tensor):
        
        # use initial_hidden state to perform first state of computations
        attn = self._attention_provider.forward(initial_hidden_state, memory)
        hidden_state = self._state_controller.forward(attn, initial_hidden_state)
        
        return hidden_state


# In[13]:


reasoner = ReasoningProcess(5, 
                            AttentionOverMemory(AttentionCoefProvider(BilinearSimilarity(300,300), True)),
                            StateController(torch.nn.GRUCell(300, 300)))


# # Seq2SeqWrapper, that returns (all_states, last_state) tuple

# In[14]:


import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.util import sort_batch_by_length, get_lengths_from_binary_sequence_mask


class PytorchLastStateSeq2SeqWrapper(Seq2SeqEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the first one as a single output.  This
    wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from ``get_output_dim``.

    In order to be wrapped with this wrapper, a class must have the following members:

        - ``self.input_size: int``
        - ``self.hidden_size: int``
        - ``def forward(inputs: PackedSequence, hidden_state: torch.autograd.Variable) ->
          Tuple[PackedSequence, torch.autograd.Variable]``.
        - ``self.bidirectional: bool`` (optional)

    This is what pytorch's RNN's look like - just make sure your class looks like those, and it
    should work.

    Note that we *require* you to pass sequence lengths when you call this module, to avoid subtle
    bugs around masking.  If you already have a ``PackedSequence`` you can pass ``None`` as the
    second parameter.
    """
    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        super(PytorchLastStateSeq2SeqWrapper, self).__init__()
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        try:
            is_bidirectional = self._module.bidirectional
        except AttributeError:
            is_bidirectional = False
        return self._module.hidden_size * (2 if is_bidirectional else 1)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:

        if mask is None:
            return self._module(inputs, hidden_state)[0]
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(inputs,
                                                                                           sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.data.tolist(),
                                                     batch_first=True)

        # Actually call the module on the sorted PackedSequence.
        packed_sequence_output, state = self._module(packed_sequence_input, hidden_state)

        # Deal with the fact the LSTM state is a tuple of (state, memory).
        if isinstance(state, tuple):
            state = state[0]

        # Restore the original indices and return the final state of the
        # top layer. Pytorch's recurrent layers return state in the form
        # (num_layers * num_directions, batch_size, hidden_size) regardless
        # of the 'batch_first' flag, so we transpose, extract the relevant
        # layer state (both forward and backward if using bidirectional layers)
        # and return them as a single (batch_size, self.get_output_dim()) tensor.

        # now of shape: (batch_size, num_layers * num_directions, hidden_size).
        unsorted_state = state.transpose(0, 1).index_select(0, restoration_indices)

        # Extract the last hidden vector, including both forward and backward states
        # if the cell is bidirectional. Then reshape by concatenation (in the case
        # we have bidirectional states) or just squash the 1st dimension in the non-
        # bidirectional case. Return tensor has shape (batch_size, hidden_size * num_directions).
        try:
            last_state_index = 2 if self._module.bidirectional else 1
        except AttributeError:
            last_state_index = 1
        last_layer_state = unsorted_state[:, -last_state_index:, :]

        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
        # Restore the original indices and return the sequence.
        return unpacked_sequence_tensor.index_select(0, restoration_indices), last_layer_state.contiguous().view([-1, self.get_output_dim()])


# In[15]:


from typing import Type

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.augmented_lstm import AugmentedLstm
#from allennlp.modules.seq2seq_encoders.intra_sentence_attention import IntraSentenceAttentionEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
#from allennlp.modules.seq2seq_encoders.pytorch_last_state_seq2seq_wrapper import PytorchLastStateSeq2SeqWrapper
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm

class Seq2SeqWrapperWithLastState:
    """
    For :class:`Registrable` we need to have a ``Type[Seq2SeqEncoder]`` as the value registered for each
    key.  What that means is that we need to be able to ``__call__`` these values (as is done with
    ``__init__`` on the class), and be able to call ``from_params()`` on the value.

    In order to accomplish this, we have two options: (1) we create a ``Seq2SeqEncoder`` class for
    all of pytorch's RNN modules individually, with our own parallel classes that we register in
    the registry; or (2) we wrap pytorch's RNNs with something that `mimics` the required
    API.  We've gone with the second option here.

    This is a two-step approach: first, we have the :class:`PytorchSeq2SeqWrapper` class that handles
    the interface between a pytorch RNN and our ``Seq2SeqEncoder`` API.  Our ``PytorchSeq2SeqWrapper``
    takes an instantiated pytorch RNN and just does some interface changes.  Second, we need a way
    to create one of these ``PytorchSeq2SeqWrappers``, with an instantiated pytorch RNN, from the
    registry.  That's what this ``_Wrapper`` does.  The only thing this class does is instantiate
    the pytorch RNN in a way that's compatible with ``Registrable``, then pass it off to the
    ``PytorchSeq2SeqWrapper`` class.

    When you instantiate a ``_Wrapper`` object, you give it an ``RNNBase`` subclass, which we save
    to ``self``.  Then when called (as if we were instantiating an actual encoder with
    ``Encoder(**params)``, or with ``Encoder.from_params(params)``), we pass those parameters
    through to the ``RNNBase`` constructor, then pass the instantiated pytorch RNN to the
    ``PytorchSeq2SeqWrapper``.  This lets us use this class in the registry and have everything just
    work.
    """
    PYTORCH_MODELS = [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]

    def __init__(self, module_class: Type[torch.nn.modules.RNNBase], return_last_state=False) -> None:
        self._return_last_state = return_last_state
        self._module_class = module_class

    def __call__(self, **kwargs) -> PytorchSeq2SeqWrapper:
        return self.from_params(Params(kwargs))

    def from_params(self, params: Params) -> PytorchSeq2SeqWrapper:
        if not params.pop('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        if self._module_class in self.PYTORCH_MODELS:
            params['batch_first'] = True
        module = self._module_class(**params.as_dict())
        if not self._return_last_state:
            return PytorchSeq2SeqWrapper(module)
        else:
            return PytorchLastStateSeq2SeqWrapper(module)


# In[16]:


Seq2SeqEncoder.register("l_lstm")(Seq2SeqWrapperWithLastState(torch.nn.LSTM, return_last_state=True))


# # Reasonet model

# In[17]:


from allennlp.common import Params
import pyhocon


# In[18]:


config = """
{
  "dataset_reader": {
    "type": "squad",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "character_tokenizer": {
          "byte_encoding": "utf-8",
          "start_tokens": [259],
          "end_tokens": [260]
        }
      }
    }
  },
  "train_data_path": "./squad/train-v1.1.json",
  "validation_data_path": "./squad/dev-v1.1.json",
  "model": {
    "type": "reasonet_dev",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "./glove/glove.6B.100d.txt.gz",
        "embedding_dim": 100,
        "trainable": false
      },
      "token_characters": {
        "type": "character_encoding",
        "embedding": {
          "num_embeddings": 262,
          "embedding_dim": 16
        },
        "encoder": {
          "type": "cnn",
          "embedding_dim": 16,
          "num_filters": 100,
          "ngram_filter_sizes": [5]
        },
        "dropout": 0.2
      }
    },
    "num_highway_layers": 2,
    "state_controller": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "phrase_layer": {
      "type": "l_lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "similarity_function": {
      "type": "linear",
      "combination": "x,y,x*y",
      "tensor_1_dim": 200,
      "tensor_2_dim": 200
    },
    "modeling_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 800,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.2
    },
    "dropout": 0.2
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["passage", "num_tokens"], ["question", "num_tokens"]],
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 100,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+em",
    "cuda_device": 0,
    "learning_rate_scheduler":  {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2,

    },
    "no_tqdm": true,
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    }
  }
}
"""


# In[19]:


params = Params(pyhocon.ConfigFactory.parse_string(config))


# In[20]:


params


# # Reasonet model

# In[21]:


import logging
from typing import Any, Dict, List, Optional

from torch.nn.functional import nll_loss

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import util
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.regularizers import RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
# Should import this, but can't:
#from allennlp.training.metrics import SquadEmAndF1
from allennlp.modules.attention import Attention
from allennlp.modules.similarity_functions.bilinear import BilinearSimilarity
from allennlp.nn.util import weighted_sum
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# In[22]:


from typing import Tuple

from overrides import overrides

from allennlp.common import squad_eval
from allennlp.training.metrics.metric import Metric


@Metric.register("squad")
class SquadEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD
    evaluation script.
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    @overrides
    def __call__(self, best_span_string, answer_strings):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        exact_match = squad_eval.metric_max_over_ground_truths(
                squad_eval.exact_match_score,
                best_span_string,
                answer_strings)
        f1_score = squad_eval.metric_max_over_ground_truths(
                squad_eval.f1_score,
                best_span_string,
                answer_strings)
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0


# In[23]:


@Model.register("reasonet_dev")
class Reasonet(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    attention_similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    evaluation_json_file : ``str``, optional
        If given, we will load this JSON into memory and use it to compute official metrics
        against.  We need this separately from the validation dataset, because the official metrics
        use all of the annotations, while our dataset reader picks the most frequent one.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 state_controller: Seq2SeqEncoder,
                 phrase_layer: Seq2SeqEncoder,
                 attention_similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 max_timesteps: int = 10) -> None:
        #super(Reasonet, self).__init__(vocab, regularizer)
        super(Reasonet, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))

        self._state_controller = state_controller
        self._phrase_layer = phrase_layer
        self._matrix_attention = MatrixAttention(attention_similarity_function)
        self._modeling_layer = modeling_layer

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        state_controller_dim = modeling_dim

        similarity_function = CUDA_wrapper(BilinearSimilarity(state_controller_dim, modeling_dim))
        state_rnn = CUDA_wrapper(torch.nn.GRUCell(state_controller_dim, state_controller_dim))

        self.termination_gate = CUDA_wrapper(TerminationGate(state_controller_dim))

        coef_provider = AttentionCoefProvider(similarity_function, True)

        self.reasoner_step = ReasoningProcessStep(
            AttentionOverMemory(coef_provider),
            StateController(state_rnn)
        )
        self.max_timesteps = max_timesteps
        
        span_start_input_dim = 2*modeling_dim
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_input_dim = 2*modeling_dim
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` index.  If
            this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` index.  If
            this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        # We use question_last_state to initialize state controller
        question_encoding, question_last_state = self._phrase_layer(embedded_question, question_lstm_mask)
        passage_encoding, _ =  self._phrase_layer(embedded_passage, passage_lstm_mask)

        encoded_question = self._dropout(question_encoding)
        encoded_passage = self._dropout(passage_encoding)
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
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

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)

        # !!! modelled passage = M
        M = modeled_passage
        
        no_answer_yet = CUDA_wrapper(torch.ones(batch_size).byte())
        
        span_start_logits_final = CUDA_wrapper(torch.zeros(batch_size, passage_length))
        span_end_logits_final = CUDA_wrapper(torch.zeros(batch_size, passage_length))
        
        expected_reward = 0
        reasoner_last_state = question_last_state
        proceed_until_now_prob = 1.
        for step in range(self.max_timesteps):
            reasoner_last_state = self.reasoner_step.forward(reasoner_last_state, M)
            proceed_prob, stop_prob = self.termination_gate.forward(reasoner_last_state)
            
            # Shape: (batch_size, passage_length, modeling_dim)
            tiled_reasoner_last_state = reasoner_last_state.unsqueeze(1).expand(
                batch_size, passage_length, modeling_dim
            )

            # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
            answer_ready_representation = self._dropout(
                torch.cat([modeled_passage, modeled_passage * tiled_reasoner_last_state], dim=-1)
            )

            # ! Start prediction

            # Shape: (batch_size, passage_length)
            span_start_logits = self._span_start_predictor(answer_ready_representation).squeeze(-1)

            # Shape: (batch_size, passage_length)
            span_end_logits = self._span_end_predictor(answer_ready_representation).squeeze(-1)

            # ! End prediction

            may_be_answer_now = torch.bernoulli(stop_prob.data.squeeze()).byte()
            if step < self.max_timesteps - 1:
                answer_now = may_be_answer_now * no_answer_yet
            else:
                answer_now = no_answer_yet
            no_answer_yet = no_answer_yet * (1 - answer_now)
            
            span_start_logits_final += answer_now.unsqueeze(-1).float() * span_start_logits.data
            span_end_logits_final += answer_now.unsqueeze(-1).float() * span_end_logits.data

            span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
            span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
            
            if span_start is not None:
                #loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
                #loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
                reward = batchwise_index(
                    util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1)
                ) + batchwise_index(
                    util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1)
                )
                #early_answer_penalty = int(step < 1)
                early_answer_penalty = 0
                #late_answer_penalty = 1
                late_answer_penalty = 0
                if step < self.max_timesteps - 1:
                    expected_reward += torch.mean(
                        reward * stop_prob * proceed_until_now_prob * (1 + early_answer_penalty)
                    )
                else:
                    expected_reward += torch.mean(reward * proceed_until_now_prob * (1 + late_answer_penalty))
            proceed_until_now_prob = proceed_until_now_prob * proceed_prob

        span_start_logits = Variable(span_start_logits_final, requires_grad=False)
        span_end_logits = Variable(span_end_logits_final, requires_grad=False)
        
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
            
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)
        # Shape: (batch_size, passage_length)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

        best_span = self._get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span
        }
        if span_start is not None:
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            output_dict["loss"] = -expected_reward
            
        if metadata is not None:
            output_dict['best_span_str'] = []
            for i in range(batch_size):
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].data.cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
        
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        metrics = {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }
        return metrics

    @staticmethod
    def _get_best_span(span_start_logits: Variable, span_end_logits: Variable) -> Variable:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, 2).fill_(0)).long()

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidirectionalAttentionFlow':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        num_highway_layers = params.pop("num_highway_layers")
        state_controller = Seq2SeqEncoder.from_params(params.pop("state_controller"))
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
        dropout = params.pop('dropout', 0.2)

        # TODO: Remove the following when fully deprecated
        evaluation_json_file = params.pop('evaluation_json_file', None)
        if evaluation_json_file is not None:
            logger.warning("the 'evaluation_json_file' model parameter is deprecated, please remove")

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        mask_lstms = params.pop('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   num_highway_layers=num_highway_layers,
                   state_controller=state_controller,
                   phrase_layer=phrase_layer,
                   attention_similarity_function=similarity_function,
                   modeling_layer=modeling_layer,
                   dropout=dropout,
                   mask_lstms=mask_lstms,
                   initializer=initializer,
                   regularizer=regularizer)


# # Training details

# ## Preprocessing

# In[24]:


import argparse
import json
import logging
import os
import sys
from copy import deepcopy
import percache

from allennlp.common.checks import ensure_pythonhashseed_set
from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
#from allennlp.common.util import prepare_environment
from allennlp.data import Dataset, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer


# In[25]:


from itertools import zip_longest
from typing import Any, Callable, Dict, List, TypeVar, Union
import random

import torch
import numpy

from allennlp.common.checks import log_pytorch_version_info
from allennlp.common.params import Params

JsonDict = Dict[str, Any] # pylint: disable=invalid-name

def prepare_environment(params: Union[Params, Dict[str, Any]]):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.
    Parameters
    ----------
    params: Params object or dict, required.
        A ``Params`` object or dict holding the json parameters.
    """
    seed = params.pop("random_seed", 13370)
    numpy_seed = params.pop("numpy_seed", 1337)
    torch_seed = params.pop("pytorch_seed", 133)

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)


# In[26]:


from allennlp.commands.train import train_model


# In[ ]:


serialization_dir = './serialization_dir_'
cache_dir = './cache_dir_'


# In[ ]:


prepare_environment(params)

os.makedirs(serialization_dir, exist_ok=True)
sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout)  # type: ignore
sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr)  # type: ignore
handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
logging.getLogger().addHandler(handler)
serialization_params = deepcopy(params).as_dict(quiet=True)
with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
    json.dump(serialization_params, param_file, indent=4)

cache = percache.Cache(cache_dir)

# Now we begin assembling the required parts for the Trainer.
dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
train_data_path = params.pop('train_data_path')
logger.info("Reading training data from %s", train_data_path)
train_data = dataset_reader.read(train_data_path)

validation_data_path = params.pop('validation_data_path', None)
if validation_data_path is not None:
    logger.info("Reading validation data from %s", validation_data_path)
    validation_data = dataset_reader.read(validation_data_path)
    combined_data = Dataset(train_data.instances + validation_data.instances)
else:
    validation_data = None
    combined_data = train_data

vocab = cache(Vocabulary.from_params)(params.pop("vocabulary", {}), combined_data)
iterator = cache(DataIterator.from_params)(params.pop("iterator"))

cache.close()

vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

model = Model.from_params(vocab, params.pop('model'))

train_data.index_instances(vocab)
if validation_data:
    validation_data.index_instances(vocab)

trainer_params = params.pop("trainer")
trainer = Trainer.from_params(model,
                              serialization_dir,
                              iterator,
                              train_data,
                              validation_data,
                              trainer_params)
params.assert_empty('base train command')
trainer.train()

# Now tar up results
archive_model(serialization_dir)

