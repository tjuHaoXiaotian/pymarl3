REGISTRY = {}

from .hpn_rnn_agent import HPN_RNNAgent
from .hpns_rnn_agent import HPNS_RNNAgent
from .asn_rnn_agent import AsnRNNAgent
from .deepset_hyper_rnn_agent import DeepSetHyperRNNAgent
from .deepset_rnn_agent import DeepSetRNNAgent
from .gnn_rnn_agent import GnnRNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_agent import RNNAgent
from .updet_agent import UPDeT

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["hpn_rnn"] = HPN_RNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
REGISTRY["deepset_rnn"] = DeepSetRNNAgent
REGISTRY["deepset_hyper_rnn"] = DeepSetHyperRNNAgent
REGISTRY["updet_agent"] = UPDeT
REGISTRY["asn_rnn"] = AsnRNNAgent
REGISTRY["gnn_rnn"] = GnnRNNAgent
