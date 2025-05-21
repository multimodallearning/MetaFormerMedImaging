from datasets.med_mnist_dataset import MedMNISTDataModule
from datasets.nnunet_data_aug import nnUNetDataAugmentation2D
from matplotlib import pyplot as plt

def plot_img(img, title):
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.figure()
    plt.imshow(img, 'gray')
    plt.title(title)
    plt.axis('off')

data_aug = nnUNetDataAugmentation2D(True)
gray_ds = MedMNISTDataModule('OrganCMNIST', batch_size=1, spatial_size=224, use_data_aug=False)
color_ds = MedMNISTDataModule('PathMNIST', batch_size=1, spatial_size=224, use_data_aug=False)
gray_ds.setup('test')
color_ds.setup('test')

for ds in [gray_ds, color_ds]:
    img, _ = next(iter(ds.test_dataloader()))
    plot_img(img, 'original')

    gaus_img = data_aug._random_gaussian_noise(img, std=0.1, p=1)
    plot_img(gaus_img, 'gaussian noise')

    min_bright_img = data_aug._random_brightness(img, 0.75, 1)
    plot_img(min_bright_img, 'min brightness')

    max_bright_img = data_aug._random_brightness(img, 1.25, 1)
    plot_img(max_bright_img, 'max brightness')

    min_contrast_img = data_aug._random_contrast(img, 0.75, 1)
    plot_img(min_contrast_img, 'min contrast')

    max_contrast_img = data_aug._random_contrast(img, 1.25, 1)
    plot_img(max_contrast_img, 'max contrast')

    gamma_img = data_aug._random_gamma2d(img, (0.75, 1.5), False, 1e-7, True, 1)
    plot_img(gamma_img, 'gamma')

    inv_gamma_img = data_aug._random_gamma2d(img, (0.75, 1.5), True, 1e-7, False, 1)
    plot_img(inv_gamma_img, 'inv gamma')

    low_res_img = data_aug._random_simulate_low_resolution(img, 0.5, 1)
    plot_img(low_res_img, 'low res')

    high_res_img = data_aug._random_simulate_low_resolution(img, 1., 1)
    plot_img(high_res_img, 'high res')

plt.show()

