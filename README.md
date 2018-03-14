# OPTIMIZING THE LATENT SPACE REPRESENTATION

This is an implementation of https://arxiv.org/abs/1707.05776. There are two branches in this repository.

master branch: ![equation](https://latex.codecogs.com/gif.latex?\inline&space;$l_2$) as loss function

Laplace branch: Laplacian pyramid loss as loss function.

# Why 2 different loss functions?

According to the paper, ![equation](https://latex.codecogs.com/gif.latex?\inline&space;$l_2$) loss is supposed to render blurry images from learned noise representations. And laplacian pyramid loss is supposed to generated sharper images. My experience has been the opposite. The generated images through ![equation](https://latex.codecogs.com/gif.latex?\inline&space;$l_2$) loss are much sharper than Laplacian Pyramid Loss.

Before discussing possible reasons, let me explain what Laplacian pyrmaids and how to construct one: 
## Laplacian Pyramid:
Build a Gaussian kernel to convolve the image with.

1. Call your original Image $G_0$
> A gaussian kernel is used to introduce blur into the image, also called Gaussian blur. See https://en.wikipedia.org/wiki/Gaussian_blur for more information.
2. Convolve $G_0$ with Gaussian Kernel and call the resulting image $G_1$.
> $G_1$ will now have a reduced shape. 
> For example: with 3x3 Gaussian kernel and 28x28 image, the size of $G_1$ is 24x24.
3. Expand $G_1$ and call the resulting image $G_1exp$.
> $G_0$ and $G_1exp$ should have same dimensions.
> I have implemented this expand as conv_transpose with the same Gaussian Kernel.
4. First level of Laplacian Pyramid:
> $L_1$ = $G_0$ - $G_1exp$
5. Sub-sample the image and repeat from step 1.
> Drop alternate rows and columns.

## Where it went wrong?

I have implemented Step 3(expand) by doing a conv_transpose with the same Gaussian Kernel. May be this is where it went wrong. 
Future commits in Laplace banch will focus on correcting this algorithm. For now, I intend to follow tutorial at https://www.cs.utah.edu/~arul/report/node12.html.

## Overview:
A brief overview of how the model woks:
Each real image from the dataset is paired with a random noise. Lets call this random noise $z_i$ for image $x_i$. For all 60,000 images in the dataset we will have 60,000 pairs of ($z_i$, $x_i$).
> From now on let call this 60,000 as n, that is the no. of images in dataset.

Let Z be a vector of dimensions $n$ x $d$ where d is the size of noise vector. And $Z_{n,d}$ contains all n noise vectors. So, the 4th row of $Z$ is the noise vector corresponding to 4th image in the dataset.

To select individual noise vectors out of this $Z$ we use a one_hot_vector $O_{1,n}$.
Let $N_i$ be a noise vector corresponding to some image i.
> $N_{1,d}$ = $O_{1,n}$ x $Z_{n,d}$
> where $i^{th}$ element in $O$ is 1 and rest are 0. Matrix Multiplication will select $i^{th}$ row from $Z$.

Reshape $N_i$ of size(1,d) as (p,p,p,1). where d = $p^3$. Now we can use this 4-D tensor to perform conv_transpose.

The no.of layers to generate image will depend on:
1. The size of noise vector.
2. The choice of padding and strides.    

Call this generated image $x'$. Calulcate $l_2$ loss between $x'$ and $x_i$. Using backprop, update the noise vector. 

In the initial stages, generated images will be blurry and distorted, but after just 5 or 10 passes through dataset, generated images are sharper and resemble real images from dataset.


## Generated Images:
![Image generated from noise](/images/generated/run2/10360.png)
You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.
