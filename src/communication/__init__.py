from .gnn import GCNComm, GATComm

REGISTRY = {}

REGISTRY["gcn"] = GCNComm
REGISTRY["gat"] = GATComm