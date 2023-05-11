from enum import IntEnum


class Specialization(IntEnum):
    None_ = 0
    MurTree = 1


class LowerBound(IntEnum):
    None_ = 0
    Similarity = 1


class Branching(IntEnum):
    None_ = 0
    Dynamic = 1


class CacheInit(IntEnum):
    None_ = 0
    Dynamic = 1
    FromUser = 2


class Heuristic(IntEnum):
    None_ = 0
    InformationGain = 1
    InformationGainRatio = 2
    GiniIndex = 3


class DiscrepancyStrategy(IntEnum):
    None_ = 0
    Incremental = 1
    Double = 2


class FitMethod(IntEnum):
    MurTree = 0
    InfoGain = 1


class DataStructure(IntEnum):
    Raw = (0,)
    Horizontal = (1,)
    Bitset = (2,)
    ReversibleBitset = (3,)
