#  Weather LLM Web App 
The LLM Weather App is a Streamlit application that leverages Language Model (LLM) technology to provide up-to-date weather information. This app dynamically generates prompts based on user inquiries to fetch detailed weather reports and generates corresponding weather images.


## Features
* Fetches and displays detailed weather reports.
* Generates weather images based on the weather report.
* Supports multiple tools for fetching weather data (e.g., SerpAPI, OpenWeatherMap).
* User-configurable model and tool selection.

## Prerequisites
* Python 3.8+
* API keys for the selected weather data tool (SerpAPI or OpenWeatherMap) and Google Cloud (for VertexAI).

## Building the App
### Setup
To build the Streamlit App, you must you must first install Streamlit
* https://docs.streamlit.io/get-started/installation

Since we are using VertexAI, you must have a Google Cloud account with Vertex AI APIs enabled and gcloud CLI
* https://cloud.google.com/
* https://cloud.google.com/sdk/gcloud

Once you have set up your Google Cloud Project, you can sign in through the command line with
```
gcloud auth login
```
The Streamlit app has several requirements that must be installed
```
pip install -r /path/to/requirements/requirements.txt
```
The *.py* script must also be updated to reflect the path to your google cloud credentials .json as well as your PROJECT_ID, BUCKET_URI, and chosen REGION.

Now, if all paths and credentials are valid, when running the script you will be advised to run the following in the command line 
```
streamlit run /path/to/script/Weather_App.py
```
You should now see Streamlit User Interface appear in your browser at a localhost port.

*  Screenshot of Streamlit User Interface 
![app_fp](https://github.com/E-C-T/Gen-AI/blob/main/LLM_Weather_App/app_fp.jpg?raw=true)
* A screenshot of the output of the question “Hows the wether it n seattle ?”
![app_demo](https://github.com/E-C-T/Gen-AI/blob/main/LLM_Weather_App/app_demo.jpg?raw=true)


# Model Assessments
In order to run this notebook Google Cloud Credentials and an OpenAI API key must be provided.

The LLM models were assessed utilizing FLASK for Fine-Grained Skills Assessment over the following categories
  - Robustness
  - Correctness
  - Efficiency
  - Factuality
  - Commonsense
  - Comprehension
  - Insightfulness
  - Completeness
  - Metacognition
  - Readability
  - Conciseness
  - Harmlessness

The Image Generation Model Accuracy was evaluated through the combination of providing a weather_image_prompt and weather_image that is generated from the prompt to OpenAI's 'gpt-4-vision-preview'. The evaluation on the similarity between the image prompts and images is then passed to Google Vertex AI's 'gemini-pro' to provide a score between 0 and 5, where 0 indicates low similarity and 5 indicates high similarity.  

## LLM Models 
*LLM_Evaluation_Using_FLASK.ipynb*
  - gemini-pro
  - text-bison@001
  - text-bison@002
![LLM_Eval](https://github.com/E-C-T/Gen-AI/blob/main/LLM_Weather_App/Fine_Grained_Assesment_LLM_models.png?raw=true)


## Image Generation Model
*Weather_App.ipynb*
  - imagegeneration@005

Initializing prompts using a list of cities:

```
city_list = [
    "Seattle", "Los Angeles", "San Francisco", "Chicago", "New York", "Boston", 
    "Miami", "Houston", "Dallas", "Atlanta", "Washington D.C.", "Philadelphia", 
    "Las Vegas", "Denver", "Portland", "San Diego", "Phoenix", "Minneapolis", 
    "Orlando"
]
```
The following scores were obtained:

```
scores = [3.5, 5, 4.5, 4.8, 5, 0, 4.5, 4.5, 3.5, 0, 1.5, 4.5, 4, 4.5, 4, 1.5, 5, 0, 0, 4.5]
```

Average Score:

```
3.24
```

Note: This evaluation is imperfect and depends highly on many pieces working together well including: the model generating the image prompt, the model generating the image, the model performing the evaluation, the model scoring the evaluation, and the quality of all the prompts used. Future work for Image Generation Model evaluation might be better suited through the use of a CLIP model and measuring the distance be the embedded image_prompt and embedded image in the latant space through use of a similarity matrix.





