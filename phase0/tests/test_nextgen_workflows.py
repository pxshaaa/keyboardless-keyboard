import json
import numpy as np
import pytest
import torch
from phase0.analysis.task_context import build
from phase0.analysis.streaming_replay import ranges,replay


def test_context_preserves_unsupported_and_binds_source(tmp_path):
    f=tmp_path/'working.txt'
    f.write_text('fetchUser fetchUser für get2germany')
    result=build([f])
    assert result['terms']==['fetch','user']
    assert set(result['unsupported_terms'])=={'für','get2germany'}
    assert len(result['sources'][0]['sha256'])==64
    with pytest.raises(ValueError):
        build([f],0)


@pytest.mark.parametrize('n',[1,29,30,31,60,61,700])
def test_stream_chunks_cover_every_output_once(n):
    indices=[]
    for lo,hi,a,b in ranges(n):
        assert 0<=lo<=a<b<=hi<=n
        indices.extend(range(a,b,2))
        assert hi<=b+30
    assert indices==list(range(0,n,2))


def test_stream_never_uses_future_landmarks():
    class Model:
        def __call__(self,f,lens):
            signal=f[0,::2,:29]+f.mean()*torch.arange(29)
            return signal.log_softmax(-1)[None]
    rng=np.random.default_rng(4)
    points=rng.normal(size=(240,2,21,3)).astype(np.float32)*10+100
    changed=points.copy();changed[180:]+=1000
    t=np.arange(240)/60
    first,events,_=replay(Model(),points,t)
    second,_,_=replay(Model(),changed,t)
    np.testing.assert_array_equal(first[:75],second[:75])
    assert len(first)==120
    assert events[-1]['final']


def test_pixel_branch_has_gradient_and_geometry_control():
    from phase0.analysis.typing_pixels import PixelResidual
    torch.set_num_threads(1)
    torch.manual_seed(0)
    m=PixelResidual()
    torch.nn.init.normal_(m.temporal[-1].weight,std=.01)
    rgb=torch.rand(8,6,24,24,requires_grad=True)
    geo=torch.randn(8,506);base=torch.randn(8,29).log_softmax(-1)
    m(rgb,geo,base,True)[:,1].sum().backward()
    assert rgb.grad.abs().sum()>0
    a=m(rgb,geo,base,False)
    b=m(torch.zeros_like(rgb),geo,base,False)
    torch.testing.assert_close(a,b)


def test_causal_features_prefix_invariance():
    from phase0.analysis.causal_student import causal_features
    p=np.random.default_rng(7).normal(size=(60,2,21,3)).astype(np.float32)*10+100
    short=causal_features(p[:30])
    full=causal_features(p)
    torch.testing.assert_close(short,full[:15],rtol=0,atol=0)


def test_causal_student_chunk_state_matches_full_pass():
    from phase0.analysis.causal_student import Student
    torch.manual_seed(0)
    model=Student().eval()
    x=torch.randn(1,40,506)
    with torch.inference_mode():
        full,_=model(x)
        first,state=model(x[:,:17])
        second,_=model(x[:,17:],state)
    torch.testing.assert_close(full,torch.cat([first,second],1),atol=1e-6,rtol=1e-6)
