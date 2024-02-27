import matplotlib.pyplot as plt
from typing import List
def plot_losses(losses: List[float], output_file: str) -> None:
	epochs = range(1, len(losses) + 1)
	plt.plot(epochs, losses, marker='o', linestyle='-', color='blue', label='Loss')
	plt.title('Loss Over Time')
	plt.xlabel('Interval')
	plt.ylabel('Avg Interval Loss')
	plt.legend()
	plt.grid(True)
	
	plt.savefig(output_file, format='pdf')