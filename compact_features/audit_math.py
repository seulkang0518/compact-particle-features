"""Fixed-reference Laplace MMD diagnostics, separate from stopped Riesz training."""
import torch
import torch.nn.functional as F


def perturb_latents(x, kind, seed=42):
    if kind == 'clean':
        return x.clone()
    if kind == 'noise':
        g = torch.Generator(device=x.device).manual_seed(seed)
        return x + .15 * torch.randn(x.shape, generator=g, device=x.device)
    if kind == 'blur':
        return F.avg_pool2d(F.pad(x, (1,1,1,1), mode='reflect'), 3, stride=1)
    if kind == 'shuffle':
        n,c,h,w=x.shape
        if h % 4 or w % 4:
            raise ValueError('Shuffle needs spatial dimensions divisible by 4')
        patches=x.reshape(n,c,h//4,4,w//4,4).permute(0,2,4,1,3,5).reshape(n,-1,c,4,4)
        g=torch.Generator().manual_seed(seed)
        order=torch.randperm(patches.shape[1], generator=g).to(x.device)
        return patches[:,order].reshape(n,h//4,w//4,c,4,4).permute(0,3,1,4,2,5).reshape_as(x)
    raise ValueError(kind)


def reference_scales(reference):
    scales={}
    for k,r in reference.items():
        d=torch.pdist(r.float())
        scales[k]=d.median().clamp_min(1e-6).detach()
    return scales


def mmd2(features, reference, scales, tau=1.):
    """Biased V-statistic; scales fitted on independent real references only."""
    terms=[]
    for key,x in features.items():
        y=reference[key].detach().float(); x=x.float()
        scale=scales[key]*tau
        kernel=lambda a,b: torch.exp(-torch.cdist(a,b)/scale)
        terms.append(kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean())
    return torch.stack(terms).mean()
