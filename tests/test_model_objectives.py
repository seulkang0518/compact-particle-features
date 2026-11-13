import pytest
import torch
from torch import nn

from compact_features.model import CompactEncoder, ReconstructionModel, collect_features, masked_reconstruction_loss
from compact_features.objectives import feature_mse, geometry_loss


torch.set_num_threads(2)


def test_encoder_shapes_retained_heads_and_per_image_normalization():
    torch.manual_seed(4)
    channels=dict(zip(('conv1','layer1','layer2','layer3','layer4'),(6,8,10,12,14)))
    model=CompactEncoder(width=4,blocks=1,teacher_channels=channels).eval()
    x=torch.randn(3,4,32,32,requires_grad=True)
    maps=model(x)
    assert model.channels==channels
    for (name,value),size in zip(maps.items(),(32,32,16,8,4)):
        assert value.shape==(3,channels[name],size,size)
        torch.testing.assert_close(value[:1],model(x[:1])[name],rtol=2e-5,atol=2e-6)
    sum(v.square().mean() for v in maps.values()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(head.weight.grad is not None for head in model.adapters.values())


def test_feature_collection_uses_common_positions_and_full_map_moments():
    value=torch.arange(16,dtype=torch.float32).reshape(2,2,2,2).requires_grad_()
    actual=collect_features({'stage':value},{'stage':torch.tensor([3,0])})
    torch.testing.assert_close(actual['stage/spatial'],value.flatten(2)[:,:,[3,0]].permute(2,0,1))
    torch.testing.assert_close(actual['stage/mean'][0],value.mean((-1,-2)))
    torch.testing.assert_close(actual['stage/std'][0],(value.var((-1,-2),unbiased=False)+1e-6).sqrt())
    actual['stage/spatial'].sum().backward()
    assert (value.grad.flatten(2)[:,:,1:3]==0).all()
    with pytest.raises(ValueError,match='outside'):
        collect_features({'stage':value},{'stage':[4]})


def test_feature_mse_teacher_is_detached_and_amplitude_is_preserved():
    teacher={'a':torch.randn(2,5,8,requires_grad=True),'b':torch.randn(1,5,4,requires_grad=True)}
    student={k:(2*v.detach()).requires_grad_() for k,v in teacher.items()}
    loss=feature_mse(student,teacher)
    assert loss.item()>0.9  # Independent unit normalization would make this zero.
    loss.backward()
    assert all(v.grad is None for v in teacher.values())
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in student.values())


def test_geometry_scale_invariance_and_teacher_detachment():
    torch.manual_seed(3)
    teacher={'stage':torch.randn(3,7,8,requires_grad=True)}
    student={'stage':(teacher['stage'].detach()*3).requires_grad_()}
    loss,metrics=geometry_loss(student,teacher,queries=3)
    assert abs(loss.item())<1e-5
    torch.testing.assert_close(metrics['student_distance_scale'],3*metrics['teacher_distance_scale'])
    loss.backward()
    assert teacher['stage'].grad is None
    assert torch.isfinite(student['stage'].grad).all()


def test_geometry_query_and_reference_permutations_and_self_exclusion():
    torch.manual_seed(9)
    student={'x':torch.randn(2,6,5,requires_grad=True)}
    teacher={'x':torch.randn(2,6,9,requires_grad=True)}
    loss,metrics=geometry_loss(student,teacher,queries=2,temperatures=(.2,.7))
    assert metrics['effective_neighbors_negative'].item()==pytest.approx(1.)
    permutation=torch.tensor([1,0,5,3,2,4])
    actual,_=geometry_loss({k:v[:,permutation] for k,v in student.items()},
                           {k:v[:,permutation] for k,v in teacher.items()},2,temperatures=(.2,.7))
    torch.testing.assert_close(loss,actual,rtol=2e-5,atol=2e-6)
    loss.backward()
    assert torch.isfinite(student['x'].grad).all() and student['x'].grad.abs().sum()>0
    assert teacher['x'].grad is None
    with pytest.raises(ValueError,match='at least two'):
        geometry_loss(student,teacher,1)
    with pytest.raises(ValueError,match='Degenerate'):
        geometry_loss({'x':torch.zeros(2,6,5)},teacher,2)


def test_geometry_gradient_respects_differentiable_scale_invariance():
    torch.manual_seed(51)
    value=torch.randn(3,7,5,requires_grad=True)
    teacher={'x':torch.randn(3,7,8)}
    loss,_=geometry_loss({'x':value},teacher,3,temperatures=(.2,.6))
    gradient=torch.autograd.grad(loss,value)[0]
    # L(a*x)=L(x) for positive a, hence its radial directional derivative is 0.
    # Detaching the student's scale gives the same forward value but violates
    # this gradient identity, which is important for feature pretraining.
    assert gradient.abs().sum()>0
    assert abs((gradient*value).sum().item())<2e-5


def test_masked_reconstruction_only_scores_masked_coordinates_and_keeps_input():
    # Identity predicts visible coordinates perfectly and masked ones as zero.
    # Thus masked MSE=1; an erroneous full-image mean would equal mask fraction.
    x=torch.ones(2,4,8,8)
    loss,detail=masked_reconstruction_loss(nn.Identity(),x,mask_ratio=.01,patch_size=4,
                                         generator=torch.Generator().manual_seed(3),return_details=True)
    torch.testing.assert_close(loss,torch.tensor(1.))
    assert detail['masked_fraction']>0
    torch.testing.assert_close(x,torch.ones_like(x))
    model=ReconstructionModel(width=4,blocks=1)
    inputs=torch.randn(2,4,32,32)
    loss=masked_reconstruction_loss(model,inputs,generator=torch.Generator().manual_seed(1))
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.stem[0].weight.grad.abs().sum()>0
    assert model(inputs).shape==inputs.shape
