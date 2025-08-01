---
layout: page
title: Unlocking Facial Recognition Systems with Master-Key Images
description: Generalized Attacks on Face Verification Systems
img: assets/img/oneface.png
importance: 3
category: Ongoing
---




We have invested a considerable amount of time into this project. Once we have completed all the necessary steps, I will provide a comprehensive update on this page, including project details, core concepts, algorithms, and outcomes. I will also publicize the code on <a href="https://github.com/enazari" style=" text-decoration: underline;">my GitHub</a>. For the time being, here I will share both early and recent experiment results. You can also find the final set of the experiments and results on  <a href="https://arxiv.org/abs/2309.05879" style=" text-decoration: underline;">arXiv:2309.05879</a>


<h2>Problem Definition</h2>

Face verification is the process of determining if two facial images belong to the same individual.

Face Recognition systems have not only surpassed human performance in Face verification, but have also advanced to the extent that these technologies are now employed in various public safety applications, such as border control procedures, as well as commercial applications like unlocking smartphones.  

The objective of this project was to create an image or a collection of images that could serve as master keys to unlock a Facial Recognition system. The effectiveness of a master key image is determined by its ability to unlock multiple locks. In this context, a lock refers to a face recognition system, such as the one found on smartphones.


<h3>Idea One</h3>
In the early attempts, we could generate such images that can fool a soft-max based neural network used for Face Verification task with a high success rate.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/1.png"  class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/2.png"  class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/3.png" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/4.png" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/5.png" class="img-fluid rounded z-depth-1" %}
    </div>

</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/6.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/7.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/8.png" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/9.png"  class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/10.png" class="img-fluid rounded z-depth-1" %}
    </div>

</div>
<div class="caption">
    Each of the ten images presented here is generated using a modified version of our base model. Each image functions as a master key, capable of successfully unlocking a significant percentage of the locks.
</div>


<h3>Idea Two</h3>
By implementing an other idea, we could generate such images that can fool a soft-max based neural network used for Face Verification task with even a higher success rate.


<div class="row">
    <div class='container'>
    <div class="col-lg-4 col-md-4 col-sm-8 mx-auto">
        {% include figure.html path="assets/img/DodgePersonation/second1.png"  class="img-fluid rounded z-depth-1" %}
    </div>
        </div>

</div>
<div class="caption">
    Utilizing a completely distinct concept from the previous approach, we have generated an image that achieves the highest success rate thus far.
</div>

<h3>Idea Two - Incorporating a Different Loss Function</h3>
By changing the loss function of our model, we generated other nonsensical but instersting images:
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/secondc1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/secondc2.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/secondc3.png" class="img-fluid rounded z-depth-1" %}
    </div>

</div>
<div class="caption">
    These images are generated by minimizing a different loss function than the previous one. Here we used Cosine Distance.
</div>


<h3>Idea Three</h3>
Still, we have not yet been able to generate a convincing human face. This poses a problem as it would not pass the initial step in a Facial Recognition System, thus rendering the attack ineffective. In such systems, like a smartphone's facial recognition feature that utilizes the phone's camera, the first stage involves extracting the face from the captured image. Unfortunately, the previously generated images are unable to pass this face detection process due to their lack of recognizable human facial features. As a result, our next attempt focused on generating images that not only function as master keys but are also detected as faces by the face detector. In the following section, we present an endeavor to address this issue with a new approach:


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/third1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/third2.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/third3.png" class="img-fluid rounded z-depth-1" %}
    </div>

</div>
<div class="caption">
    The outcomes of an alternative method aimed at generating master key images resembling human faces.
</div>


However, based on preliminary results, it appears that our attempts with this idea have not produced promising outcomes in terms of both the face detection step and their effectiveness as master keys.


<h3>Idea Four</h3>
Next, we tried an other idea, that inputs a human face into the system, and generates an image that has the master key propery we want. In the following, I used my face image as the input, and successfully generated an image that can open more than 50% of the locks, effectively acting as a master key:

<div class="row">
    <div class="col-lg-5 col-md-4 col-sm-4 mx-auto">
        {% include figure.html path="assets/img/DodgePersonation/fourth1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-lg-5 col-md-4 col-sm-4 mx-auto">
        {% include figure.html path="assets/img/DodgePersonation/fourth2.png" class="img-fluid rounded z-depth-1" %}
    </div>


</div>
<div class="caption">
The image on the left is the original image, while the image on the right is generated based on the left image to serve as a master key.
</div>

<h3>Idea Five</h3>
After conducting tests on softmax loss-based neural networks, we applied our methods to triplet loss-based neural networks. Unfortunately, the initial testing of these methods did not produce satisfactory results for triplet loss-based neural networks. This led us to explore another idea to address this issue. It took considerable effort and time, but eventually, we successfully cracked a triplet loss-based neural network for face verification with a success rate of 58%. This represents a significant improvement compared to the previous work, which only achieved a success rate of 43%. I invite you to take a look at the set of ten images below, where nine of them are modified images of me that can act as master keys. Can you spot the original image among these master-key images?



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000000.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000001.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000002.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000003.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/ehsan_cluster.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>

</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000004.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000005.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000006.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000007.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/DodgePersonation/cluster_000008.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>

</div>
<div class="caption">
    These are the outcomes of our latest concept. These nine images, working together, can successfully unlock 58% of the locks. In comparison, previous approaches could only unlock 43% of the locks using the same number of generated images.
</div>


<h3>The Full Study on arXiv</h3>
As I highlighted earlier, I've presented only the preliminary experiment results here. For the complete methodology and final experiments and results, please refer to arXiv:

<div class='container'>
  <div class="publications">
    <p><strong>Unlocking Facial Recognition Systems with Master-Key Images</strong><br>
    <a href="https://arxiv.org/abs/2309.05879" style="text-decoration: underline;">arXiv:2309.05879</a></p>
  </div>
</div>

