Here is the complete `README.md` file for the **IntuiqAI** project, generated using the provided structure and constraints.

-----

# IntuiqAI: Intelligent Data-to-Dashboard MVP

**Tagline:** Accelerate your data journey from raw CSV to a complete, AI-driven dashboard and presentation script using Python and Streamlit.

-----

## Table of Contents

1.  [Introduction](https://www.google.com/search?q=%23introduction)
2.  [Key Features](https://www.google.com/search?q=%23key-features)
3.  [Project Architecture](https://www.google.com/search?q=%23project-architecture)
4.  [Prerequisites](https://www.google.com/search?q=%23prerequisites)
5.  [Local Installation & Setup](https://www.google.com/search?q=%23local-installation--setup)
6.  [How to Run the App](https://www.google.com/search?q=%23how-to-run-the-app)
7.  [Usage Workflow](https://www.google.com/search?q=%23usage-workflow)
8.  [Technologies Used](https://www.google.com/search?q=%23technologies-used)

-----

## Introduction

**IntuiqAI** is a Minimum Viable Product (MVP) designed to automate the initial, often complex stages of data analysis and visualization. It bridges the gap between uploading a raw dataset and presenting a business-ready dashboard. By leveraging **Ollama** and the **Llama 3.2:1b** model, the application provides AI-powered suggestions for data cleaning, dashboard planning, and storytelling, all within an intuitive **Streamlit** user interface.

## Key Features

  * **CSV Data Ingestion & Analysis:** Load any CSV, automatically detect data types (numerical, categorical), count missing values, and detect outliers using the IQR method.
  * **Guided Data Cleaning:** Based on the analysis, the app suggests cleaning steps (imputation for missing values, capping for outliers) for user review and approval.
  * **SQL Storage:** The cleaned dataset is persistently stored in a local **SQLite** database using SQLAlchemy.
  * **AI-Powered Storytelling:** The user provides a **Domain** and **Purpose**, and the AI suggests a coherent storyline, key insights, and dashboard layout.
  * **Interactive Streamlit Dashboard:** Renders an interactive, PowerBI-like dashboard with AI-recommended elements, including graphs (Plotly/Seaborn), filters, and KPI cards.
  * **Presentation Script Generation:** An option to generate a detailed, ready-to-use script and guide for presenting the generated dashboard.
  * **Dataset Query Chatbot:** A dedicated AI chatbot for answering user questions about the specific dataset and dashboard context.

-----

## Project Architecture

The project is intentionally split into two core files for clear separation of duties:

  * **`app.py`:** Contains the main Streamlit application logic. This file manages the UI, defines the multi-step workflow, handles user interactions, and maintains state persistence using `st.session_state`.
  * **`utils.py`:** Serves as the backend library. It houses all helper functions for core tasks: data analysis, data cleaning logic, managing the SQLite database, and all interactions/prompts with the Ollama API.

-----

## Prerequisites

Before running the application, ensure you have the following installed locally:

1.  **Python 3.10+**
2.  **Ollama:** The large language model runtime must be installed and running in the background.
3.  **Llama 3.2:1b Model:** You must have the specific LLM model pulled locally.

To pull the required model, run this command in your terminal:

```bash
ollama pull llama3.2:1b
```

-----

## Local Installation & Setup

Follow these steps to get IntuiqAI running on your local machine.

### 1\. Clone the Repository

```bash
git clone [YOUR_REPO_URL]
cd IntuiqAI
```

*(Replace `[YOUR_REPO_URL]` with your actual repository address.)*

### 2\. Install Python Packages

IntuiqAI requires several libraries for data handling, UI, and visualization.

```bash
pip install streamlit pandas numpy sqlalchemy ollama plotly seaborn
```

### 3\. Verify Ollama

Ensure the Ollama server is active and the `llama3.2:1b` model is available locally before starting the Streamlit app.

-----

## How to Run the App

Start the IntuiqAI application using the Streamlit command line tool:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser.

-----

## Usage Workflow

The application guides the user through a clear, multi-step process:

1.  **Upload:** Upload a single **CSV** file using the file uploader.
2.  **Analyze & Approve:** Review the detailed analysis of data types, missing values, and outliers. Approve the recommended data cleaning steps.
3.  **Clean & Store:** The data is automatically cleaned and stored in the local SQLite database.
4.  **Define Purpose:** Input the **Domain** (e.g., 'Finance', 'E-commerce') and **Purpose** (e.g., 'Sales Trend Analysis', 'Customer Segmentation').
5.  **View Dashboard:** The AI suggests and renders the interactive dashboard (graphs, KPIs, filters) based on your inputs.
6.  **Insights:** Choose to interact with the **AI Chatbot** for data queries or generate the **Presentation Script** for the final presentation.

-----

## Technologies Used

  * **Python:** Core programming language.
  * **Streamlit:** For building the reactive, interactive web user interface.
  * **Pandas & NumPy:** For robust data analysis and manipulation.
  * **Ollama & Llama 3.2:1b:** For all AI/LLM-driven tasks (analysis verification, storytelling, script generation, and chatbot).
  * **SQLAlchemy & SQLite:** For persistent, local data storage.
  * **Plotly & Seaborn:** For generating the interactive, high-quality dashboard visualizations.
