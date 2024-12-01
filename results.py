import matplotlib.pyplot as plt

def generate_plot(log, path):
   plt.figure(figsize=(12, 8))  # Larger figure
   epochs, d_losses, g_losses = [], [], []
   
   for line in log:
       line = line.rstrip(";\n")
       e, d, g = line.split(";")
       epochs.append(int(e))
       d_losses.append(float(d))
       g_losses.append(float(g))

   plt.grid(True, alpha=0.3)  # Add light grid
   plt.plot(epochs, d_losses, label='Discriminator Loss', linewidth=2)
   plt.plot(epochs, g_losses, label='Generator Loss', linewidth=2)
   
   plt.xlabel('Epoch', fontsize=12)
   plt.ylabel('Loss', fontsize=12)
   plt.tick_params(axis='both', labelsize=10)
   plt.legend(fontsize=10, loc='upper right')
   
   plt.tight_layout()  # Adjust spacing
   plt.savefig(path + "/plot.png", dpi=300, bbox_inches='tight')
   plt.show()
