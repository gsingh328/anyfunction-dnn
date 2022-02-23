import numpy as np
import matplotlib.pyplot as plt


folding_factor = np.asarray([2, 4, 8, 16])
accuracy_no_pe = np.asarray([90.54, 90.61, 90.32, 90.40])
accuracy_pe = np.asarray([90.38, 90.17, 90.00, 90.04])

plt.plot(folding_factor, accuracy_no_pe, 'o-', color='blue', label="No PE")
plt.plot(folding_factor, accuracy_pe, 'o-', color='green', label="With PE")
plt.axhline(y=90.91, color='red', linestyle='-', label='Baseline')
plt.xscale('log', base=2)
plt.legend()
plt.title('Results with Folded Middle Layers')
plt.xlabel('Folding Factor')
plt.ylabel('Accuracy')
plt.ylim(85., 92.)
plt.savefig("accuracy.png", dpi=400)
