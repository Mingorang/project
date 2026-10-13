import numpy as np
from week02.generators import iid_paths
from week05.score import similarity
def test_seed(): assert np.array_equal(iid_paths([.01,-.01],2,4,7),iid_paths([.01,-.01],2,4,7))
def test_score(): assert similarity([1,2],[1,2],1)["similarity"]==1
