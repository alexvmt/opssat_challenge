# opssat_challenge

Competition: https://kelvins.esa.int/opssat/

Data: https://zenodo.org/record/6524750

Starter kit: https://gitlab.com/EuropeanSpaceAgency/the_opssat_case_starter_kit

Key questions:

- How to best augment images, i. e. increase amount of images?
	- Create new augmented copies of original images for each type of augmentation?
	- Create new augmented copies of original images using multiple/all augmentations?
- What's the best validation strategy?
	- No validation set?
	- Validation set without augmented images?
	- Validation set with augmented images? If yes, same augmentations as for training set or just a subset of them?
	- Manual or automatic split?
- Usually it's a good idea to normalize images before training a neural net. Normalization, however, cannot be applied to the test set in this case since it's hidden. Does it still make sense to normalize the images before training?
- More generally, we cannot do test-time augmentation since the test set is hidden.
- There seem to be issues with reproducibility when using a GPU for training.
