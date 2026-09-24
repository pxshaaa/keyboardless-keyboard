import cv2
import numpy as np
import torch
from phase0.analysis.motion_residual import track,MotionResidual


def test_flow_recovers_known_translation():
    rng=np.random.default_rng(5)
    image=cv2.GaussianBlur(rng.integers(0,256,(128,128),dtype=np.uint8),(3,3),0)
    moved=cv2.warpAffine(image,np.float32([[1,0,2],[0,1,-1]]),(128,128))
    pts=np.array([[x,y] for x in [32,48,64,80] for y in [32,48,64,80]],np.float32)
    flow,good=track(image,moved,pts)
    assert good.sum()>=14
    np.testing.assert_allclose(flow[good],np.tile([2,-1],(good.sum(),1)),atol=.15)


def test_residual_starts_as_unchanged_baseline_and_backpropagates():
    torch.manual_seed(1)
    model=MotionResidual();base=torch.randn(40,29).log_softmax(-1)
    result=model(torch.randn(40,506),torch.randn(40,210),base)
    torch.testing.assert_close(result,base)
    (-result[:,2].mean()).backward()
    assert model.out.weight.grad.abs().sum()>0


def test_full_resolution_flow_recovers_translation():
    rng=np.random.default_rng(7)
    image=cv2.GaussianBlur(rng.integers(0,256,(256,256),dtype=np.uint8),(3,3),0)
    moved=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,-2]]),(256,256))
    pts=np.array([[x,y] for x in [64,96,128,160] for y in [64,96,128,160]],np.float32)
    flow,good=track(image,moved,pts,resolution=1.)
    assert good.sum()>=14
    np.testing.assert_allclose(flow[good],np.tile([4,-2],(good.sum(),1)),atol=.2)


def test_full_decoder_rejects_missing_scores(monkeypatch):
    import pytest
    from phase0.analysis.motion_full import select_pool
    from phase0.analysis import v4_suggest
    monkeypatch.setattr(v4_suggest,'pool_scores',lambda p,r:np.array([0.]))
    p=dict(texts=['a'],base=np.array([True]),isfz=np.array([False]),gen_rank={},lm={'8b':np.array([np.nan]),'p05b':np.array([0.])})
    with pytest.raises(ValueError,match='Unscored'):
        select_pool(p,dict(prompt='p7',K=5))
