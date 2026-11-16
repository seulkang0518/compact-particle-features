import torch
from compact_features.audit_math import perturb_latents, reference_scales, mmd2
from compact_features.audit_math import normalized_descent, recovery_metrics, paired_feature_errors


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


def test_paired_feature_control_exact_loss_gradient_and_pairing():
    # Different branch widths, nonconstant targets, and unequal scales make
    # coordinate averaging or accidental population matching detectable.
    x={'a':torch.tensor([[3.,4.],[2.,0.]],requires_grad=True),
       'b':torch.tensor([[5.],[1.]],requires_grad=True)}
    target={'a':torch.tensor([[1.,0.],[0.,2.]],requires_grad=True),
            'b':torch.tensor([[1.],[3.]],requires_grad=True)}
    scales={'a':torch.tensor(2.,requires_grad=True),'b':torch.tensor(4.,requires_grad=True)}
    errors=paired_feature_errors(x,target,scales)
    torch.testing.assert_close(errors,torch.tensor([3.,1.125]))
    errors.mean().backward()
    torch.testing.assert_close(x['a'].grad,torch.tensor([[.25,.5],[.25,-.25]]))
    torch.testing.assert_close(x['b'].grad,torch.tensor([[.125],[-.0625]]))
    assert all(t.grad is None for t in target.values())
    assert all(s.grad is None for s in scales.values())
    swapped={k:v.flip(0) for k,v in target.items()}
    assert not torch.allclose(errors,paired_feature_errors(x,swapped,scales))

    clean=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
    loss=paired_feature_errors({'f':clean},{'f':clean},{'f':torch.tensor(2.)}).mean()
    gradient,=torch.autograd.grad(loss,clean)
    assert loss==0 and torch.count_nonzero(gradient)==0
    torch.testing.assert_close(normalized_descent(clean,gradient,.01),clean)
