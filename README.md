# OPS-SAT Challenge

Competition: https://kelvins.esa.int/opssat/

Data: https://zenodo.org/record/6524750

Starter kit: https://gitlab.com/EuropeanSpaceAgency/the_opssat_case_starter_kit

Here are the provided tiles for the mountain class to get an idea of what the images look like:

![mountains](mountains.png 'mountains')

Key questions and observations:

- How to best augment images, i. e. increase amount of images?
- What's the best validation strategy?
- Usually it's a good idea to normalize images before training a neural net. Normalization, however, cannot be applied to the test set in this case since it's hidden. Does it still make sense to normalize the images before training?
- More generally, we cannot do test-time augmentation since the test set is hidden.
- There seem to be issues with reproducibility when using a GPU for training.
