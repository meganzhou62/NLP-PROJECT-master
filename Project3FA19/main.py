from ffnn1fix import main as ffnn_main


FLAG = 'FFNN'
number_of_epochs = 50


def main():
	if FLAG == 'FFNN':
		hidden_dim = 32
		ffnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)



if __name__ == '__main__':
	main()
