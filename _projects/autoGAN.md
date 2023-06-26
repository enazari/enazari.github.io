---
layout: page
title: AutoGAN 
description: An Automated Human-out-of-the-Loop Approach for Training Generative Adversarial Networks
img: assets/img/publication_preview/mfid.png
importance: 2
category: Published
---

<h3>Problem Definition</h3>

Training a GAN can be quite complex compared to standard learning algorithms. Unlike optimizing an objective function, GAN training involves a competitive game where one player (the generator) tries to maximize an objective function while the other player (the discriminator) aims to minimize it. This presents a challenge in finding the Nash equilibrium, which is the potential solution to this game. <strong> Determining when to stop training a GAN is also a significant challenge</strong>, as there is no universally accepted criterion. Researchers often rely on visual inspection or quantitative evaluation metrics like Inception Score (IS) and Fréchet Inception Distance (FID). However, these approaches have limitations, such as the subjective nature of visual inspection and the need for pre-trained models for quantitative evaluation. <strong>This project aims to address these challenges and provide an automatic and systematic approach for determining the optimal point to stop GAN training across different data types.</strong> The code for this project is publicly available on <a href="https://github.com/enazari/autoGAN" style=" text-decoration: underline;">my GitHub</a>.

<h3>The Solution</h3>

We introduce an algorithm called AutoGAN, which aims to automate the training of GANs without the need for human intervention. Traditional GAN training can be challenging, requiring human involvement and subjective visual inspection. Moreover, most evaluation metrics are designed for image datasets, limiting the application of GANs to non-image data. AutoGAN addresses these issues by utilizing an oracle, a customizable scoring mechanism that assesses the quality of generated samples. 

An oracle is a function that assesses a given generator and assigns a score that reflects the quality of the generated samples. The definition of a "better" sample depends on the specific oracle instance chosen by the end-user. For instance, in some cases, a "better" sample might closely match the dataset's distribution. However, in other scenarios, it could indicate increased diversity in the data or, in the case of images, higher quality and sharpness. 

The algorithm iteratively trains the GAN and evaluates its performance based on an oracle's scores. AutoGAN allows for temporary performance dips, giving the GAN the opportunity to recover and reach optimal results. Once no further improvement is observed for a specified number of iterations, AutoGAN returns the best-trained GAN model. This algorithm minimizes human intervention and is applicable to various data types, including tabular and image data. Extensive experiments demonstrate the superiority of AutoGAN over GANs trained with manual visual inspection.



<h3>An Oracle Instance</h3>
FID oracle instance utilizes a truncated version of the InceptionNet model. Initially, the InceptionNet model is trained on the ImageNet dataset. Then, a GAN is trained on the target real dataset, and the generator component of the GAN generates a specific number of fake samples. Both the real and fake data are processed by the truncated InceptionNet model to obtain their respective InceptionNet-represented features. These features are used to calculate the Fréchet Inception Distance (FID) score. A lower FID score indicates higher quality outputs from the GAN.
<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/autogan/fid.png" title="An oracle instance based on FID score" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    An oracle instance based on FID score
</div>

<h3>A Glimpse into Results</h3>
In the figure bellow, a set of the comparative results of using a Conditional GAN trained via different methods, used for Oversampling of an imbalanced binarized dataset based on fashion-MNIST classes 2 and 4 is shown:
<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/autogan/results.png" title="Oversampling results of an imbalanced binarized dataset based on fashion-MNIST classes 2 and 4" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    The objective is to classify between class 2 and class 4 within an imbalanced fashion-MNIST dataset. The 'initial' data point signifies the outcomes obtained from classifying the imbalanced dataset, while the 'Manual' data point indicates the results obtained by classifying a balanced dataset. The balanced dataset was generated using a GAN, and the training process was manually monitored to determine the appropriate stopping point.
</div>

In the figure above, all the points, except for the ones labeled as 'initial' and 'Manual,' represent different runs of AutoGAN using various Oracle instances. We can observe that both the 'Manual' method and the AutoGAN method show similar improvements in the minority class after oversampling. This suggests that we have achieved comparable results to a GAN that is manually inspected, using an automated system that operates without human intervention.

<h3>The Potentials of the Solution</h3>
Exploring the transferability of image-to-image translation techniques to tabular-to-tabular translation or applying image denoising methods to tabular data denoising could be interesting avenues for future research.


<h3>The Published Results</h3>
To gain a comprehensive understanding of the AutoGAN algorithm and access the extensive results from a wide range of experiments, I kindly refer you to our research paper:
<div class='container' >
<div class="publications">

  {% bibliography -f {{ site.scholar.bibliography }} -q @*[number={{4}}]* %}

</div>
</div>

<!-- <div class='container' >
 The code for AutoGAN can be found at <a href='https://github.com/enazari/autoGAN'>https://github.com/enazari/autoGAN</a>.
</div> -->

