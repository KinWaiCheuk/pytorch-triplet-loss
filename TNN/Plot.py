import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import os

# Define our own plot function
def scatter(x, labels, root='plot', subtitle=None, dataset='MNIST'):
    print("IS IT EVEN WORKING WTF")
    if dataset != 'EMNIST':
        num_classes = len(set(labels)) # Calculate the number of classes
    else:
        num_classes = 27

    palette = np.array(sns.color_palette("hls", num_classes)) # Choosing color
    print(palette.shape)
    ## Create a seaborn scatter plot ##
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    ax.axis('off')
    ax.axis('tight')
    # ---------------------------- ##

    ## Add label on top of each cluster ##
    if dataset=='MNIST':
        idx2name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif dataset=='CIFAR10':
        idx2name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset=='EMNIST':
        idx2name = ['N/A','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    else:
        raise Exception("Please specify the dataset")

    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, idx2name[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    ## ---------------------------- ##

    if subtitle != None:
        plt.suptitle(subtitle)

    if not os.path.exists(root):
        os.makedirs(root)
    plt.savefig(os.path.join(root, str(subtitle)))