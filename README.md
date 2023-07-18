For Advanced Topics in Data Science class

This project is motivated by finding ways machine vision could be used to aid against the effects of climate change, specifically the worsening of forest fires. Already we use air patrols and weather monitoring, but our solution would be less costly and would require less aid from the public (via local reports). Our solution was to create a CNN model to provide wildfire detection and attempt localization.

We hit some technical problems, finding relevant and quality data lost us some time, and our use of some undocumented TensorFlow ML libraries took a lot of time and effort to understand and engage with.

In the inspiration of how bullet trains were modeled after kingfisher birds' beaks, we thought of ways to get around the problem of forest fire image quality- transfer learning. Training our model on tf_flowers dataset, a set of bright, colorful flower images in a forest background, we could then retrain the model for forest fires.

We attempted a neural architecture search using AutoKeras on the flowers dataset. Using feature extraction, transfer learn that model onto our forest fire dataset, using Tensorboard and HParams to optimize performance, outputting an EfficientNet model pretrained on ImageNet, that yielded 96% accuracy.

We ended with stating some improvements to be made on the project, including
* working with video instead of image
* localization issues (usefulness in question when some images are engulfed in flame)
* more visualizations and tensorflow library issues
