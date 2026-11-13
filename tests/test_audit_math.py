import torch
from compact_features.audit_math import perturb_latents, reference_scales, mmd2
from compact_features.audit_math import normalized_descent, recovery_metrics


def test_shuffling_preserves_values_and_seed():
    x=torch.arange(4*32*32).reshape(1,4,32,32).float()
    y=perturb_latents(x,'shuffle',42)
    assert not torch.equal(x,y)
    torch.testing.assert_close(x.flatten().sort().values,y.flatten().sort().values)
    torch.testing.assert_close(y,perturb_latents(x,'shuffle',42))
    torch.testing.assert_close(x,perturb_latents(x,'clean'))


def test_mmd_identity_scale_and_input_gradients():
    torch.manual_seed(2)
    r={'f':torch.randn(8,5)}
    s=reference_scales(r)
    assert abs(float(mmd2(r,r,s)))<1e-6
    q={'f':(torch.randn(4,5)+3).requires_grad_(True)}
    loss=mmd2(q,r,s)
    loss.backward()
    assert torch.isfinite(q['f'].grad).all() and q['f'].grad.norm()>0
    torch.testing.assert_close(loss,mmd2({'f':q['f']*2},{'f':r['f']*2},{'f':s['f']*2}))


def test_equal_budget_and_recovery_direction():
    x=torch.ones(2,1,2,2)
    clean=torch.zeros_like(x)
    gradient=x.clone(); gradient[1]*=100
    updated=normalized_descent(x,gradient,.01)
    torch.testing.assert_close(updated,torch.full_like(x,.99))
    assert recovery_metrics(x,clean,gradient)['descent_cosine_to_paired_clean'] > .999
    assert recovery_metrics(updated,clean,gradient)['paired_latent_mse'] < 1
    torch.testing.assert_close(normalized_descent(x,torch.zeros_like(x),.01),x)
    assert recovery_metrics(clean,clean,gradient)['descent_cosine_to_paired_clean'] is None
