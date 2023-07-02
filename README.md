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

### Relevant papers, blogs, and videos (first 8 important):

1. [Tutorial: Deriving the Standard Variational Autoencoder (VAE) Loss Function](https://arxiv.org/abs/1907.08956)
2. [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) (detailed paper, goes through the entire derivation; discusess SNR and score-based interpretation of diffusion models, along with classifier-free guidance)
3. [Denoising diffusion probabilistic models from first principles](https://liorsinai.github.io/coding/2022/12/03/denoising-diffusion-1-spiral.html#reverse-process) (blog, ignore the Julia code, relevant section is derivation of functional form of forward process posterior, i.e., $q(x_{t - 1}\,|\,x_t, x_0)$)
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

### tl;dr of diffusion

We train a generative model by estimating the data distribution so we can sample a new datapoint from the estimated distribution. One way to do this is to map every data point to a standard Gaussian using some process $q$, then using a neural net to learn the reverse map $p_{\theta}$. To generate a new data point, we just input Gaussian noise into $p_{\theta}$. The _forward diffusion_ process corrupts a data point by adding noise to it iteratively over $T$ timesteps. Formally, the distribution of a data point at timestep $t$ - $x_t$ - is defined as

$$q(x_t\,|\,x_{t - 1}) = \mathcal{N}\left(x_t\,;\,\mu = \sqrt{\bar\alpha_t}\,x_{t - 1},\;\sigma^2 = \sqrt{1 - \bar\alpha_t}\,I\right)$$

where $\bar\alpha_t$ and $1 - \bar\alpha_t$ approach 0 and 1 respectively as $t$ approches infinity, i.e., the distribution of the data point at the $t$th timestep approaches a standard normal distribution as $t$ approaches infinity. The forward process follows a variance schedule $\{\beta_t\}_{t=1}^{t=T}$, and we define $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{i = 1}^t\alpha_i$. We'd like to learn

$$q(x_{t - 1}\,|\,x_t) = \frac{q(x_t\,|\,x_{t - 1})\,q(x_{t - 1})}{q(x_t)} \tag{Bayes' theorem}$$

i.e., the reverse distribution, so we can start at $x_T \sim \mathcal{N}(0, 1)$ and iteratively denoise to get $x_0$ and hopefully $x_0$ is likely to be from the data distribution. One can prove that the equation for the reverse process is also a Gaussian with the following functional form

$$q(x_{t - 1}\,|\,x_t) = \mathcal{N}\left(x_{t - 1}\,;\,\mu=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\epsilon\right),\;\sigma^2=\tilde\beta_t\,I\right)$$

where $\epsilon$ is the noise that was sampled from $\mathcal{N}(0, I)$ and used to get $x_t$ using the reparametrization trick (i.e., the noise that was used to corrupt the initial data point $x_0$ to get $x_t$). $\{\tilde\beta_t\}_{t=1}^{t=T}$ is the reverse process variance schedule. Our reverse process must have the same functional form, and here we use a neural net $\epsilon_{\theta}$ to predict the noise $\epsilon$. The forward process mean is

$$\mu_q(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\epsilon\right)$$

and the reverse process mean

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\epsilon_\theta(x_t, t)\right)$$

where have substituted the $\epsilon$ with our model $\epsilon_\theta$. Since our goal is to learn the reverse process, the objective function becomes the KL divergence between the actual reverse distribution and the predicted distribution at each timestep. Both the distributions are Gaussians, and the KL divergence between two Gaussians has a closed form. Here it reduces to simply being the mean squared error between the true nose $\epsilon$ and the model predicted noise $\epsilon_{\theta}(x_t, t)$ - which is very convenient!

The training algorithm then becomes: until convergence, choose a batch of images from the dataset, choose a timestep at random for each image, sample Gaussian noise $\epsilon$ for each image (the true noise that will be added), compute $x_t$ for each image using the reparametrization trick and the sampled noise ($x_t$ is the $x$ corrupted, corrupted using $\epsilon$), compute $\epsilon_{\theta}(x_t, t)$ (the noise predicted by the model), and backprop after computing the MSE loss between true and predicted noise.

The paper [here](https://arxiv.org/abs/2208.11970) and my notes [here](https://okarthikb.github.io/site/blog/diffusion.pdf) go over all this in more detail.
