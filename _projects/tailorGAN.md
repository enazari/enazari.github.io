---
layout: page
title: TailorGAN
description: Generating Infinite-Length Images with a Relatively Small-sized Conditional Generative Adversarial Network Variant
img: assets/img/tailorGAN/t2.png
importance: 2
category: Ongoing
---

<div class='container' style='background-color: yellow; max-width: 100%;
   padding: 20px 20px 5px 20px; margin-bottom: 20px;'>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tailorGAN/celeba3.png" title="The initial results on CelebA dataset (trial 2)" class="img-fluid rounded z-depth-1" %}
    </div>

</div>
<div class="caption">
    Initial results of TailorGAN
</div>
        </div>


This project is still in its early stages, and I plan to continue it when time permits.

My primary goal is to create a GAN model with a simple and compact architecture capable of generating infinitely long images.

Here, I will only include the initial results. Once I achieve satisfactory outcomes, I will update this page with detailed information about the model.

`First, a series of tests were conducted on the MNIST dataset:`

<div class="row">
    <div class='container' style='max-width: 20%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tailorGAN/mnist.png" title="A sample of MNIST dataset (28x28 pixels)" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    A sample of MNIST dataset (28x28 pixels)
</div>

Initially, I modified a simple fully connected conditional GAN based on my idea and explored various variations using the MNIST dataset.

`Below, you will find a set of MNIST zeros, but some of the images are actually generated while the rest are from the MNIST dataset. So far, the results appear to be satisfactory:`

<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tailorGAN/zeros.png" title="a portion of the zeros is generated while the remaining part is from the MNIST dataset" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
     In the image above, a portion of the zeros is generated while the remaining part is from the MNIST dataset.
</div>



`As shown below, the initial attempts at generating infinitely long images did not yield satisfactory results:`

<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tailorGAN/zeros2.png" title="The system's ability to generalize beyond a certain threshold appears to be limited, resulting in the generation of nonsensical content." class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
     The system's ability to generalize beyond a certain threshold appears to be limited, resulting in the generation of nonsensical content.
</div>

`In the subsequent trial with a new modification to the previous model, I obtained unusual yet improved results compared to the previous ones:`

<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tailorGAN/interesting1.png" title="Despite the continued lack of coherence in the results, it is notable that they no longer exhibit a repetitive pattern as observed in the previous test." class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
     Despite the continued lack of coherence in the results, it is notable that they no longer exhibit a repetitive pattern as observed in the previous test.
</div>

`Next, I applied an other modification of the method to the CelebA dataset:`

<div class="row">
    <div class='container' style='max-width: 30%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tailorGAN/celeba.png" title="A sample of CelebA dataset (64x64 pixels)" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    A sample of CelebA dataset (64x64 pixels)
</div>

`Here are some initial results obtained:`

<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tailorGAN/celeba2.png" title="The initial results on CelebA dataset (Model A)" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    The initial results on CelebA dataset (Model A)
</div>


<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tailorGAN/celeba3.png" title="The initial results on CelebA dataset (Model B)" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    The initial results on CelebA dataset (Model B)
</div>


I plan to address the issues and provide more results in the near future.


