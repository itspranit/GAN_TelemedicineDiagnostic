import matplotlib.pyplot as plt
import os

def save_images(epoch,lr,hr,fake_hr,save_dir="logs"):
    os.makedirs(save_dir,exist_ok=True)
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    
    ax[0].imshow(lr[0])
    ax[0].set_title("LR")
    ax[0].axis("off")
    
    ax[1].imshow((hr[0]+1.0)/2.0)
    ax[1].set_title("HR")
    ax[1].axis("off")
    
    ax[2].imshow((fake_hr[0]+1.0)/2.0)
    ax[2].set_title("SR")
    ax[2].axis("off")
    
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.close()