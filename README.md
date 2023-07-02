## Denoising Diffusion Probabilistic (and Implicit) Models

<a target="_blank" href="https://colab.research.google.com/github/okarthikb/diffusion/blob/main/diffusion.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<div align='center'>
  <img src='https://github.com/okarthikb/diffusion/assets/86470305/776bebfb-8574-45a3-8ee7-88a61e681203'/>
</div>

This is an implementation of a diffusion model trained on MNIST using a ConvNeXt and attention-based U-Net (inspired by the 2020 DDPM paper) with DDIM sampling. I wanted something more concise than [Hugging Face's explainer](https://huggingface.co/blog/annotated-diffusion) on diffusion models (based on [lucidrains's implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)), so here's a (hopefully) more concise implementation. I've also linked in the notebook all the sources I went through :)

My U-Net model:

<div align='center'>
  <img src='U-Net.png'/>
</div>


The forward and reverse diffusion processes are defined in a single `Sampler` class:

```python
class Sampler:
  def __init__(self, max_t, n_step, schedule, device, eta=1, shape=(1, 28, 28)):
    self.n_step = n_step
    self.device = device
    self.eta = eta
    self.shape = shape

    abar = schedule(max_t).to(device)
    self.abar = abar / abar[0]
    self.sqrt_abar = rearrange(self.abar.sqrt(), 'l -> l 1 1 1')
    self.sqrt_bbar = rearrange((1 - self.abar).sqrt(), 'l -> l 1 1 1')

    self.tau = torch.arange(0, max_t, max_t // n_step) + 1

  # forward diffusion process sampling
  def forward(self, x, t):
    eps = torch.randn_like(x, device=self.device)
    mu_t = self.sqrt_abar.index_select(0, t) * x
    noise_t = self.sqrt_bbar.index_select(0, t) * eps
    return mu_t + noise_t, eps

  # reverse diffusion process sampling loop
  @torch.no_grad()
  def loop(self, model, n_sample=1, classes=None):
    model.eval()

    if classes is not None:
      assert n_sample == len(classes), 'n_sample must equal batch size'

    x_t = torch.randn(n_sample, *self.shape, device=self.device)
    xs = [x_t]
    tau_b = repeat(self.tau, 'l -> l n', n=n_sample).to(self.device)

    for i in range(self.n_step - 1, 0, -1):

      bbar = 1 - self.abar[self.tau[i]]
      beta = 1 - self.abar[self.tau[i]] / self.abar[self.tau[i - 1]]
      beta = (beta * (1 - self.abar[self.tau[i - 1]]) / bbar).clip(0, 0.999)
      alpha = 1 - beta

      eps = model(x_t, tau_b[i], classes)
      x_t = alpha.rsqrt() * (x_t - beta * eps * bbar.rsqrt())

      if self.eta > 0:
        noise_t = beta.sqrt() * torch.randn_like(x_t, device=self.device)
        x_t += noise_t * eta ** 0.5

      xs.append(x_t)

    return xs
```

Here, `forward_sample` does one iteration of the training algorithm above and `loop` is the whole sampling algorithm.

<div align='center'>
  <img width="922" alt="diffusion" src="https://github.com/okarthikb/Diffusion/assets/86470305/d98c9d24-b63e-4442-9826-9d9114f0e932"/>
</div>

### Relevant papers, blogs, and videos (first 8 important)

1. [Tutorial: Deriving the Standard Variational Autoencoder (VAE) Loss Function](https://arxiv.org/abs/1907.08956)
2. [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) (detailed paper, goes through the entire derivation; discusess SNR and score-based interpretation of diffusion models, along with classifier-free guidance)
3. [Denoising diffusion probabilistic models from first principles](https://liorsinai.github.io/coding/2022/12/03/denoising-diffusion-1-spiral.html#reverse-process) (blog, ignore the Julia code, relevant section is derivation of functional form of forward process posterior)
4. [Tutorial on Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://www.youtube.com/watch?v=cS6JQpEY9cs&t=2010s) (video, pretty long, just most of the contents above in video form)
5. [Proof: Kullback-Leibler divergence for the multivariate normal distribution](https://statproofbook.github.io/P/mvn-kl.html) (important proof, required to get the closed-form expression for reverse diffusion objective)
6. [DDPMs](https://arxiv.org/abs/2006.11239) (the Ho et al. paper)
7. [Improving DDPMs](https://arxiv.org/abs/2102.09672) (introduces cosine schedule and learnable variances for reverse diffusion)
8. [DDIMs](https://arxiv.org/abs/2010.02502) (nice trick to speed up diffusion inference)
9. [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) (blog by Yang Song, a good primer on score functions and score-based generative modelling; idea discovered independently from diffusion, connection with diffusion models [later realized](https://www.quantamagazine.org/the-physics-principle-that-inspired-modern-ai-art-20230105/)\)
10. [MIT 6.S192 - Lecture 22: Diffusion Probabilistic Models, Jascha Sohl-Dickstein](https://www.youtube.com/watch?v=XCUlnHP1TNM&t=1016s) (original diffusion paper author lecture)
11. [Grokking Diffusion Models](https://nonint.com/2022/10/31/grokking-diffusion-models/) (this is re various ways to look @ diffusion models)
12. [Iterative 𝛼-(de)Blending: a Minimalist Deterministic Diffusion Model](https://ggx-research.github.io/publication/2023/05/10/publication-iadb.html) (viewing diffusion models from an image processing POV)
13. [On the Theory of Stochastic Processes, with Particular Reference to Applications](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-First-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/On-the-Theory-of-Stochastic-Processes-with-Particular-Reference-to/bsmsp/1166219215.pdf) (old 1949 paper, relevant to equivalence of forward and reverse diffusion functional forms)
14. [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU&t=251s) (the video referencing the paper above)
