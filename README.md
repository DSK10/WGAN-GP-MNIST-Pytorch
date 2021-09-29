# WGAN-GP-MNIST-Pytorch
Generating Handwritten digits (MNIST) with Wasserstein GAN with gradient penalty (WGAN-GP)
Due to vanishing gradient problem in BCE (Binary cross Entropy), WGAN uses W-Loss which approximates the Earth Mover's Distance (EMD)
This solves the problem of bounded distance of BCE (between 0 and 1)
W-Loss is no longer bounded and measures distance of real and fake distribution to positive real values.
Gradient penalty is used to maintain the continuity on the Critic's NN
It is virtually not possible to check critic's gradient at each possible point of the feature space, so the gradient penalty is calculated on interpolated image.


W-loss is given by:
Min(g) and Max(c)

E(c(x)) - E(c(g(z))) + λE( l2_normalise( ∇c(X̂) ) - 1 )**2
