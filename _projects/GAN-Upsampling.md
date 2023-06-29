---
layout: page
title: Exploring GAN-Based Oversampling across Varied Data Difficulty Factors
description: This research examines the effects of data difficulty factors on GAN-based upsampling in imagery and cybersecurity data. Factors include imbalance ratio, sample size, data dimensionality, and class overlap. The study aims to understand their impact on performance.
img: assets/img/publication_preview/setting2.png
importance: 3
category: Published
---


<h3>Problem Definition</h3>
This project aims to explore how different factors in imbalanced datasets affect the performance of GAN-based upsampling techniques in both imagery and cybersecurity domains. We investigate the impact of factors like class imbalance, sample size, data dimensionality, and class overlap when using a CGAN architecture for data augmentation in imagery data tests. We conduct experiments using the MNIST image dataset, generating multiple versions with varying difficulty factors for controlled observation. For cybersecurity data, we focus on the influence of class imbalance and sample size. The code for this project is publicly available on <a href="https://github.com/enazari/GAN-upsampling" style=" text-decoration: underline;">my GitHub</a>.

To evaluate the upsampling framework, we use a 5-fold cross-validation process with two settings: the baseline setting and the CGAN-based augmentative setting. We test the CGAN-based framework on MNIST imagery datasets and non-imagery cybersecurity datasets. Additionally, we create different versions of these datasets with varying characteristics for analysis.

<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/upsampling/setting1.png" title="The baseline setting" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    The baseline setting
</div>

<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/upsampling/setting2.png" title="The CGAN-based augmentative setting" class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    The CGAN-based augmentative setting
</div>


The CGAN-based augmentative up-sampling strategy involves training a CGAN on the original imbalanced training set, incorporating both minority and majority class cases. The trained generator generates synthetic data for the minority class, which is merged with the original training set to create a balanced training set.

Performance assessment metrics, particularly the f1-score for imbalanced datasets, are computed for each setting. For a more detailed understanding, refer to the figures depicting the baseline framework and the CGAN-based up-sampling architecture.


<h3>Experiments with imagery datasets</h3>

In our experiments, we utilized the MNIST dataset, which consists of grayscale images of hand-written digits with a size of 28 pixels by 28 pixels. Our goal was to create multiple datasets with varying attributes like sample size, class overlap, imbalance ratio, and number of features. To achieve this, we followed a specific procedure.

To form the datasets, we selected instances of the digit 'four' as the majority class cases from the MNIST dataset. We chose different sample sizes (1000, 400, 200, and 100) for the majority class while maintaining the original order of the instances. We then introduced three degrees of rotation (30 degrees, 45 degrees, and 90 degrees) to generate the minority class instances. These rotations created different levels of class overlap, making it more challenging for a classifier to distinguish between them.

Additionally, we generated datasets with varying imbalance ratios (0.4, 0.2, and 0.1) for the minority class, and we resized the original images to smaller dimensions (14x14, 8x8, and 4x4 pixels) to create datasets with different numbers of features (784, 196, 64, and 16). These manipulations allowed us to study the relationship between imbalance ratio, feature count, and the effectiveness of the CGAN in addressing the imbalance issue.

By following this procedure, we were able to create diverse datasets with different attributes to explore their impact on the performance of our models.


<div class="row">
    <div class='container' style='max-width: 100%;'>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/upsampling/mnist.png" title="Illustration of generating various image sizes and rotations by applying them to an original MNIST dataset image positioned at the top left corner." class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    Illustration of generating various image sizes and rotations by applying them to an original MNIST dataset image positioned at the top left corner.
</div>




<h3>Overall Results with Imagery datasets</h3>

In our analysis, we compared the average F1-score results from 144 tests before and after applying CGAN data augmentation. We measured the F1-score for both the majority class and the minority class.

Before augmentation, the majority class had an F1-score of 0.940310, which slightly decreased to 0.912214 after augmentation. In contrast, the minority class showed a significant improvement in the F1-score, increasing from 0.434325 without augmentation to 0.701059 with augmentation.

Overall, the results indicate that CGAN data augmentation has a positive impact on improving the F1-score for the minority class. However, there is a slight decrease in the F1-score for the majority class after augmentation. The aggregated F1-scores show a 3% decrease for the majority class and a notable 26% increase for the minority class. These findings suggest that employing CGAN-based augmentative upsampling is beneficial for datasets with multiple data difficulty factors, especially for improving the performance of the minority class. 


During our project, we faced a challenge related to tabular non-imagery datasets, specifically in the context of cybersecurity datasets. Unlike imagery datasets where visual assessment of the CGAN generator's outputs is possible, it is not feasible in cybersecurity due to the nature of the data. In imagery datasets, we can observe the generated images during training to evaluate their realism and diversity. However, in cybersecurity, visual assessment is not applicable. To overcome this challenge, we conducted empirical trials and settled on a fixed training duration of 1500 iterations. This challenge served as the basis for our subsequent project, called AutoGAN. In the <a href="{{ '/projects/autoGAN' | relative_url }}" style="color: blue; text-decoration: underline;">AutoGAN Project</a>, we will introduce a method that effectively addresses this issue by providing an efficient and convenient approach to training a GAN on both imagery and tabular data.



<h3>The Published Results</h3>
The result shown above is the outcome we selected from experiments conducted on imagery datasets. We didn't stop there! We also conducted a lot of experiments on Cybersecurity datasets, which are in a tabular format. We were excited to share our findings, so we published the results in two separate papers:
<div class='container' >
<div class="publications">

  {% bibliography -f {{ site.scholar.bibliography }} -q @*[y=1]* %}

</div>
</div>

<!-- <div class='container' >
 The code for AutoGAN can be found at <a href='https://github.com/enazari/GAN-upsampling-LIDTA21'>https://github.com/enazari/GAN-upsampling-LIDTA21</a>.
</div> -->