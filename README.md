# MajorProject

# Automated Descriptive Answer Evaluation System

This project aims to automate the evaluation process of descriptive answers, providing a user-friendly web application for both professors and students.

## Features

Role-based Login:
Professors can access a dashboard to view and evaluate student answers.
Students can log in to view assigned questions and submit their answers.
Answer Evaluation:
Professors can compare student answers to model answers they provide.
A similarity score is calculated using a fine-tuned BERT model.
Keyword weightage and assigned grades offer insights into answer quality.
Secure Authentication:
Professor login credentials are securely managed by the administrator.
Student accounts require signup for enhanced security.
## Technologies Used

Backend:
Flask: Web framework for building the application.
SQLite: Lightweight database for storing student usernames and answers.
Frontend:
HTML: Structure and content of the web pages.
CSS: Styling and visual presentation of the application.
Machine Learning:
Transfer Learning: Leverages a pre-trained BERT model (bert-base-uncased) for answer evaluation.
Fine-tuning: The model is further trained on your project's dataset to enhance its evaluation accuracy.
## Getting Started

Prerequisites:

Python 3.x (https://www.python.org/downloads/)
Flask (https://flask.palletsprojects.com/)
SQLite (https://www.sqlite.org/)
A text editor or IDE of your choice
