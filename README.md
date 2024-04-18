# Dangerous Farm Insects CNN

## Project Description
This repository contains the resources and codebase for the "Dangerous Farm Insects CNN" project, which is part of the PROG74000 Applications of ML and AI course. The project involves building a convolutional neural network (CNN) to classify images of dangerous farm insects. The trained model is then utilized within a Dockerized web application, allowing users to upload images of insects and receive classification results.

## Repository Contents
- `insects_cnn.ipynb`: Jupyter notebook containing the CNN training and evaluation.
- `farm_insects/`: Directory containing the dataset images.
- Contains the trained CNN model `scary_bugs_classification.pth`.
- `app/`: Source code for the web application and backend API.
- `docker-compose.yaml`: Docker Compose file for running the web application.

## Getting Started

### Running the Notebook
To start with the notebook, ensure you have Jupyter Notebook or JupyterLab installed. You can run the notebook by navigating to the notebook's directory and running:

`jupyter notebook insects_cnn.ipynb`

This will open the notebook in your web browser where you can run the cells to train and evaluate the CNN model.

### Running the Application with Docker Compose
Make sure Docker and Docker Compose are installed on your machine. First, update the `docker-compose.yaml` file with your machine's current IP address to ensure proper network configuration. Then, start the application by navigating to the root directory of this repository and running:
`cd app` 
`docker-compose up`

This command builds the Docker container if it's the first run and starts the web application. The application will be accessible via a web browser at the specified IP address.

`localhost:3000`

## Contributors
- Rukia
- Dominic
- Thomas
- Gavin

This project is a collaborative effort for the course PROG74000 - Applications of Machine Learning and Artificial Intelligence. 

## Notes
- Ensure that your Python environment matches the requirements specified in the notebook for compatibility.
- Added the pip install within the notebook.
- The Docker Compose setup assumes a default configuration which might need adjustments based on your Docker setup.
- Docker uses a requirements file; should only have to docker-compose up to get the app running.



