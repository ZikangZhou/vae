import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def plot_reconstruction():
    imgs = [mpimg.imread('./results_pca/reconstruction_1.png'), mpimg.imread('./results_ae_1/reconstruction_200.png'),
            mpimg.imread('./results_vae_1/reconstruction_200.png'),
            mpimg.imread('./results_cvae_1/reconstruction_200.png'), mpimg.imread('./results_pca/reconstruction_2.png'),
            mpimg.imread('./results_ae_2/reconstruction_200.png'),
            mpimg.imread('./results_vae_2/reconstruction_200.png'),
            mpimg.imread('./results_cvae_2/reconstruction_200.png'), mpimg.imread('./results_pca/reconstruction_4.png'),
            mpimg.imread('./results_ae_4/reconstruction_200.png'),
            mpimg.imread('./results_vae_4/reconstruction_200.png'),
            mpimg.imread('./results_cvae_4/reconstruction_200.png'), mpimg.imread('./results_pca/reconstruction_8.png'),
            mpimg.imread('./results_ae_8/reconstruction_200.png'),
            mpimg.imread('./results_vae_8/reconstruction_200.png'),
            mpimg.imread('./results_cvae_8/reconstruction_200.png'),
            mpimg.imread('./results_pca/reconstruction_16.png'), mpimg.imread('./results_ae_16/reconstruction_200.png'),
            mpimg.imread('./results_vae_16/reconstruction_200.png'),
            mpimg.imread('./results_cvae_16/reconstruction_200.png')]

    _, axs = plt.subplots(5, 4)
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i])
        ax.set_yticks([])
        ax.set_xticks([])
    axs[0].set_title("PPCA")
    axs[1].set_title("Deep AutoEncoder")
    axs[2].set_title("vanilla VAE")
    axs[3].set_title("CVAE")

    axs[0].set_ylabel("d=1")
    axs[4].set_ylabel("d=2")
    axs[8].set_ylabel("d=4")
    axs[12].set_ylabel("d=8")
    axs[16].set_ylabel("d=16")
    plt.tight_layout()
    plt.savefig("./reconstruction.png")
    plt.show()


def plot_generation():
    imgs = [mpimg.imread('./results_pca/sample_1.png'),
            mpimg.imread('./results_ae_1/sample_200.png'),
            mpimg.imread('./results_vae_1/sample_200.png'),
            mpimg.imread('./results_cvae_1/sample_200.png'),
            mpimg.imread('./results_pca/sample_2.png'),
            mpimg.imread('./results_ae_2/sample_200.png'),
            mpimg.imread('./results_vae_2/sample_200.png'),
            mpimg.imread('./results_cvae_2/sample_200.png'),
            mpimg.imread('./results_pca/sample_4.png'),
            mpimg.imread('./results_ae_4/sample_200.png'),
            mpimg.imread('./results_vae_4/sample_200.png'),
            mpimg.imread('./results_cvae_4/sample_200.png'),
            mpimg.imread('./results_pca/sample_8.png'),
            mpimg.imread('./results_ae_8/sample_200.png'),
            mpimg.imread('./results_vae_8/sample_200.png'),
            mpimg.imread('./results_cvae_8/sample_200.png'),
            mpimg.imread('./results_pca/sample_16.png'),
            mpimg.imread('./results_ae_16/sample_200.png'),
            mpimg.imread('./results_vae_16/sample_200.png'),
            mpimg.imread('./results_cvae_16/sample_200.png')]

    _, axs = plt.subplots(5, 4, figsize=(10, 8))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i])
        ax.set_yticks([])
        ax.set_xticks([])
    axs[0].set_title("PPCA")
    axs[1].set_title("Deep AutoEncoder")
    axs[2].set_title("vanilla VAE")
    axs[3].set_title("CVAE")

    axs[0].set_ylabel("d=1")
    axs[4].set_ylabel("d=2")
    axs[8].set_ylabel("d=4")
    axs[12].set_ylabel("d=8")
    axs[16].set_ylabel("d=16")
    plt.tight_layout()
    plt.savefig("./generation.png")
    plt.show()


if __name__ == '__main__':
    plot_reconstruction()
    plot_generation()
