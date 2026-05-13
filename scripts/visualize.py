
import matplotlib.pyplot as plt

def visualize_prediction(image,
                         mask,
                         prediction):

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("MRI Slice")

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Ground Truth")

    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title("Prediction")

    plt.tight_layout()
    plt.show()
