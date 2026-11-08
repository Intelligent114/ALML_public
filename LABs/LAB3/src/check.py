"""Public examples. Run from the LAB3 root via run.py check."""
import numpy as np


def checks(student):
    X=np.array([[3.,2.],[1.,2.],[2.,2.]])
    mu,P=student.fit_pca(X,.9)
    np.testing.assert_allclose(mu,[2.,2.])
    np.testing.assert_allclose(P@P.T,[[1,0],[0,0]],atol=1e-8)
    Z=np.array([[0.],[2.],[8.],[10.]])
    C=np.array([[0.],[8.],[30.]])
    ids, new=student.lloyd_step(Z,C.copy())
    np.testing.assert_array_equal(ids,[0,0,1,1])
    np.testing.assert_allclose(new,[[1],[9],[30]])
    y=np.array([2,-1,0,-1])
    assignment=student.label_clusters(Z,C,y)
    np.testing.assert_array_equal(assignment,[2,0,0])
    perm=[2,0,1]
    np.testing.assert_array_equal(student.label_clusters(Z,C[perm],y),np.asarray(assignment)[perm])
    return {"mean":np.asarray(mu).tolist(),"projector":(P@P.T).tolist(),
            "old_assignments":np.asarray(ids).tolist(),"updated_centers":np.asarray(new).tolist(),
            "cluster_classes":np.asarray(assignment).tolist(),"permutation_checked":True}
