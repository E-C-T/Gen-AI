# ! pip install --quiet --upgrade --user langchain langchain-core langchain-google-vertexai google-search-results google-cloud-aiplatform

import os
import requests
import streamlit as st
import pandas as pd

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

#from langchain_google_vertexai import VertexAI
from langchain.llms import VertexAI
from langchain import PromptTemplate
from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=""

PROJECT_ID = "" # use your project id
REGION = "us-central1"  #
BUCKET_URI = f""  # create your own bucket

vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)


def run_agent(agent, prompt):
    return agent.run(prompt)

    
def get_reformatted_inquiry(user_input, llm):
    """
    Processes the user input to reformat it into a structured weather inquiry.
    
    Args:
        user_input (str): The raw input string from the user.
        llm (VertexAI): The language model used for processing the input.
    
    Returns:
        tuple: Contains the validity of the inquiry, reformatted inquiry, location, and date.
    """
    response_schemas = [
        ResponseSchema(
            name="valid_inquiry",
            description="A boolean describing whether or not the user input is a valid inquiry"
        ),
        ResponseSchema(
            name="reformatted_inquiry",
            description="This is your response, a reformatted inquiry asking about the weather."
        ),
        ResponseSchema(
            name="location",
            description="The location of the inquiry."
        ),
        ResponseSchema(
            name="date",
            description="The date of the inquiry."
        ),
    ]

    # Initialize the output parser with the new response schemas
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
    You will be given a poorly formatted string from a user. First decide if this is a valid inquiry containing a Location and request for the weather.
    Then, reformat the user input as an inquiry for a search engine asking what the weather in the specified location is either for current, today, or a specified date.
    Make sure all the words are spelled correctly, including country, city, and state names, and dates.
    Provide the reformatted inquiry as well as the location and the specific date described. 
    The inquiry should request information from a search engine on the weather including the sky description, temperature information, 
    precipitation information, humidity information, and wind information. 

    {format_instructions}

    # % USER INPUT:
    {user_input}

    YOUR RESPONSE:   
    - Valid Inquiry:
    - Reformatted Inquiry:
    - Location:
    - Date:
 
    """
    prompt = PromptTemplate(
        input_variables=["user_input"],
        partial_variables={"format_instructions": format_instructions},
        template=template,
    )

    # Generate the parsed output
    promptValue = prompt.format(user_input=user_input)
    llm_output = llm(promptValue)
    parsed_output = output_parser.parse(llm_output)
    
    valid_inquiry = parsed_output['valid_inquiry']
    reformatted_inquiry = parsed_output['reformatted_inquiry']
    location = parsed_output['location']
    date = parsed_output['date']

    return valid_inquiry, reformatted_inquiry, location, date

def get_agent_output(tools, prompt, llm, weather_inquiry):
    """
    Executes an agent with a dynamically generated prompt based on user input and handles potential errors.

    This function initializes an agent with the specified tools and language model, then attempts to run
    the agent using a prompt formatted with the user's weather inquiry. It captures and handles various
    exceptions that may arise during execution, such as HTTP errors, connection issues, and timeouts,
    providing specific feedback for each type of error.

    Args:
        tools (list[Tool]): A list of tools (e.g., SerpAPIWrapper, OpenWeatherMapAPIWrapper) that the agent
                            can use to fetch weather data or perform related tasks.
        prompt (PromptTemplate): An instance of PromptTemplate used to generate the formatted prompt
                                 for the agent based on the user's weather inquiry.
        llm (LLM): The language model to be used by the agent for processing the inquiry and generating responses.
        weather_inquiry (str): The user's raw input string containing their weather-related inquiry.

    Returns:
        dict: The output from the agent's execution if successful. This output is expected to be parsed
              by another function to extract meaningful weather information. If an error occurs during
              execution, feedback is provided via Streamlit's error messaging, and an empty dictionary is returned.

    Note:
        This function uses Streamlit's `st.error` to display error messages directly in the app's interface,
        making it specific to applications built with Streamlit. For non-Streamlit environments or for more
        generic error handling, consider modifying the error handling mechanism to suit the target environment.
    """
    
    agent = initialize_agent(tools=tools, llm=llm, agent="chat-zero-shot-react-description", verbose=True)
    agent_output = {}
    
    try: 
        agent_output = agent.run(prompt.format_prompt(weather_inquiry=weather_inquiry))
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("Authentication error: Please check your API key.")
        elif e.response.status_code == 403:
            st.error("Authorization error: You don't have permission to access this resource.")
        elif e.response.status_code == 429:
            st.error("Rate limit exceeded: You've made too many requests too quickly. Please try again later.")
        else:
            st.error(f"HTTP error occurred: {e}")
            
    except requests.exceptions.ConnectionError:
        st.error("Connection error: Unable to connect to the server. Please check your internet connection.")
        
    except requests.exceptions.Timeout:
        st.error("Timeout error: The server is taking too long to respond. Please try again later.")
        
    except ValueError as e:
        st.error(f"Value error: {e}. Please check your inputs.")
        
    except TypeError as e:
        st.error(f"Type error: {e}. There might be an issue with the types of data being used.")
        
    except Exception as e:
        # Catch-all for any other unexpected exceptions
        st.error(f"An unexpected error occurred: {e}. Please try again or contact support if the issue persists.")

    return agent_output

def get_weather_info(weather_inquiry, tool_selection, llm):
    """
    Generates a detailed weather report based on a specified weather inquiry using a selected tool and a language model.

    This function dynamically generates a prompt based on the user's weather inquiry, utilizes a specified tool
    (e.g., SerpAPI or OpenWeatherMap) for fetching weather data, and leverages a language model to interpret and
    format the weather data into a structured report. The report includes details such as sky description,
    temperature, precipitation, humidity, and wind information.

    Args:
        weather_inquiry (str): The user's query about the weather, which could include location and date.
        tool_selection (str): The tool selected for fetching weather data. Valid options include "SerpAPI" and "OpenWeatherMap".
        llm (LLM): The language model used for processing the weather inquiry and generating the structured weather report.

    Returns:
        dict: A dictionary containing structured information about the weather report, including:
              - weather_report (str): A detailed description of the weather.
              - sky_description (str): Description of the sky conditions.
              - high_temperature (str): The high temperature for the specified location and date.
              - low_temperature (str): The low temperature for the specified location and date.
              - temperature_units (str): Units of measurement for the temperature.
              - precipitation (str): Information about precipitation.
              - humidity (str): The relative humidity.
              - wind_speed (str): The wind speed.
              - wind_direction (str): The wind direction.

    The function initializes an agent with the selected tool for fetching weather data and constructs a prompt template
    for the language model to generate a detailed weather report based on the weather inquiry. It uses structured
    output parsing to extract and format specific details from the model's response into a structured dictionary.
    """
    response_schemas = [
        ResponseSchema(
            name="weather_report",
            description="This is a detailed explanation of the weather including a description of the sky, high temperature, low temperature, and chance of rain."
        ),
        ResponseSchema(
            name="sky_description",
            description="This is a description of the sky, cloudy or clear the sky is for the specified location and date."
        ),
        ResponseSchema(
            name="high_temperature",
            description="This is the high temperatures for the specified location and date."
        ),
        ResponseSchema(
            name="low_temperature",
            description="This is the low temperature for the specified location and date."
        ),
        ResponseSchema(
            name="temperature_units",
            description="This is the units of measurement for the temperature for the specified location and date."
        ),
        ResponseSchema(
            name="precipitation",
            description="This is the information about precipitation, such as rain, snow, sleet, ice, hail, and drizzle for the specified location and date."
        ),
        ResponseSchema(
            name="humidity",
            description="This is the relative humidity for the specified location and date."
        ),
        ResponseSchema(
            name="wind_speed",
            description="This is the wind speed for the specified location and date."
        ),    
        ResponseSchema(
            name="wind_direction",
            description="This is the wind direction for the specified location and date."
        ),                  
    ]

    # Initialize the output parser with the new response schemas
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Get format instructions for the new response schemas
    format_instructions = output_parser.get_format_instructions()

    template = """
    {weather_inquiry}. 
    Please provide a detailed explanation of the weather including the sky description, temperature information, 
    precipitation information, humidity information, and wind information. The High temperature and the low temperature are very important.

    {format_instructions}

    % USER INPUT:
    {weather_inquiry}

    YOUR RESPONSE:
    - Weather Report:
    - Sky Description:
    - High Temperature:
    - Low Temperature:
    - Temperature units:
    - Precipitation:
    - Humidity:
    - Wind Speed:
    - Wind Direction:
    """
    
    prompt = PromptTemplate(
        input_variables=["weather_inquiry"],
        partial_variables={"format_instructions": format_instructions},
        template=template,
    )

    # Define tools for LLM weather agent
    tools = []
    if tool_selection == "SerpAPI":
        search = SerpAPIWrapper()  
        tools.append(Tool(name="Search", func=search.run, description="Useful for answering current event questions."))
        
    elif tool_selection == "OpenWeatherMap":
        tools.append(load_tools(["openweathermap-api"], llm))

    # Initialize agent and process output
    agent_output = get_agent_output(tools, prompt, llm, weather_inquiry)
        
    if agent_output:
        try:
            weather_info = output_parser.parse(agent_output)
            
        except Exception as parse_error:
            st.error(f"Error parsing agent output: {parse_error}")
            weather_info = None
            
    else:
        weather_info = None           
         
    return weather_info


def get_weather_image_prompt(weather_report, location, llm):
    """
    Creates a prompt for generating an image based on the weather report and location.
    
    Args:
        weather_report (str): The detailed weather report.
        location (str): The location of the weather inquiry.
        llm (VertexAI): The language model used for creating the image prompt.
    
    Returns:
        str: The prompt for image generation.
    """
    # Weather Image Prompt Generation
    prompt_template_weather_image = PromptTemplate(
        input_variables=['weather_report', 'location'],
        template="""
        {weather_report}. Please create an image generation prompt for a scene describing this weather report in {location}.
        Please include a cartoon character in the foreground of the scene enjoying the weather. The overall style of the image being consistent.
        """
    )

    # Initialize LLM weather chain
    llm_weather_image_chain = LLMChain(llm=llm, prompt=prompt_template_weather_image)

    weather_image_prompt = llm_weather_image_chain({'weather_report': weather_report,
                                                    'location': location})['text']
    
    return weather_image_prompt


def main_weather_logic(tool_selection, model_selection):
    """
    Orchestrates the logic for fetching and displaying weather information based on user input.
    
    This function initializes the specified language and image generation models, collects user input for a weather inquiry,
    processes this inquiry to fetch weather information, and finally, generates an accompanying weather image. The weather
    information and image are then displayed to the user.
    
    Args:
        tool_selection (str): The name of the tool selected by the user for fetching weather data. 
                              This should match one of the predefined tool names that the application supports.
        model_selection (str): The name of the language model selected by the user. This name is used to 
                               initialize the VertexAI language model for processing weather inquiries and generating 
                               responses.
    
    The function performs the following steps:
    1. Initializes the VertexAI language model and an image generation model based on the user's selections.
    2. Collects a weather inquiry from the user via text input.
    3. If the user submits an inquiry (either by pressing the "Get Weather Information" button or by entering text 
       and pressing Enter), the function proceeds to process this inquiry:
       a. It reformats the inquiry for clarity and structure using the `get_reformatted_inquiry` function.
       b. Fetches detailed weather information using the `get_weather_info` function, which leverages the selected tool.
       c. Generates a prompt for a weather image based on the fetched weather report.
    4. Displays the fetched weather report to the user.
    5. Generates and displays a weather image based on the generated prompt.
    
    The weather report includes detailed information such as the sky description, temperature, precipitation, humidity, 
    and wind. The accompanying image provides a visual representation of the reported weather conditions.
    """

    # Initialize models
    llm = VertexAI(
        model_name=model_selection,
        max_output_tokens=512,
        temperature=0.5,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )

    generation_model = ImageGenerationModel.from_pretrained("imagegeneration@005")

    # User input for weather inquiry
    user_input = st.text_input("Enter your weather inquiry:", "")

    # Process inquiry and display results
    if st.button("Get Weather Information") or user_input:
        with st.spinner('Retrieving weather information...'):
            valid_inquiry, reformatted_inquiry, location, date = get_reformatted_inquiry(user_input, llm)
            
            if valid_inquiry:       
                st.subheader(f"{location}, {date}")
                weather_info = get_weather_info(reformatted_inquiry, tool_selection, llm)
                
                if weather_info:
                    try:
                        weather_report = weather_info["weather_report"]
                        st.write(weather_report)
                        
                        # Create a DataFrame for the rest of the weather info
                        weather_details = {key: [val] for key, val in weather_info.items() if key != "weather_report"}
                        df_weather_details = pd.DataFrame.from_dict(weather_details)
                        
                        # Optionally, you might want to reorder or select specific columns to display
                        columns_order = ["sky_description", "high_temperature", "low_temperature", "temperature_units", 
                                        "precipitation", "humidity", "wind_speed", "wind_direction"]
                        df_weather_details = df_weather_details[columns_order]  
                        
                        # Display the DataFrame in Streamlit
                        st.dataframe(df_weather_details) 
                        
                        weather_image_prompt = get_weather_image_prompt(weather_report, location, llm)
                        
                    except:
                        st.error("Weather Report could not be retrieved.")
                        weather_image_prompt = None
                else:
                    st.error("Weather Report could not be retrieved. ")
            else:
                st.error("Your input may be missing a location or is unrelated to weather. Please enter your weather query again.")
            

        with st.spinner('Generating weather image...'):
            try:
                weather_image = generation_model.generate_images(prompt=weather_image_prompt)
                resized_image = weather_image.images[0]._pil_image.resize((400, 400))

                st.image(resized_image, caption=f"{location}")
            except:
                st.error("Weather Image was unable to be generated.")
                
    return
            
            
            
# Streamlit App 
st.set_page_config(page_title="Weather Information App", layout="wide")
st.title("LLM Weather App")

st.subheader("Information Provided")
st.markdown("""
<style>
.column {
    float: left;
    width: 25.00%;
    padding: 5px;
}

/* Clear floats after the columns */
.row:after {
    content: "";
    display: table;
    clear: both;
}

/* Style the list */
ul {
    list-style-type: disc; /* To keep the bullet style */
    padding-left: 20px; /* Adjust as needed for alignment */
}
</style>

<div class="row">
  <div class="column">
    <ul>
      <li>Weather Report</li>
      <li>Sky Description</li>
    </ul>
  </div>
  
  <div class="column">
    <ul>
      <li>High Temperature</li>
      <li>Precipitation</li>
    </ul>
  </div>
  
  <div class="column">
    <ul>
      <li>Low Temperature</li>
      <li>Wind Speed</li>
    </ul>
  </div>
  
  <div class="column">
    <ul>
      <li>Temperature Units</li>
      <li>Humidity</li>
    </ul>
  </div>
  
</div>
""", unsafe_allow_html=True)


# Streamlit Sidebar Setup
with st.sidebar:
    st.title("Configuration")
    
    # Tool selection    
    model_selection = st.selectbox("Choose your LLM model:", ["", "gemini-pro","text-bison@001", "text-bison@002",])  # Placeholder model names
    tool_selection = st.selectbox("Choose a tool to use:", ["", "SerpAPI", "OpenWeatherMap"])

    if tool_selection:
        api_key = st.text_input(f"Enter {tool_selection} API Key")
        
        # Check if API key is entered and initialize the tool
        if api_key:
            if tool_selection == "SerpAPI":
                serpapi_api_key = api_key
                if serpapi_api_key:
                    st.success("Key Added! The key's validity will be confirmed upon first use.") 
                    os.environ["SERPAPI_API_KEY"] = serpapi_api_key
                    
                else:
                   st.warning("The API key you entered was empty") 
                   
            elif tool_selection == "OpenWeatherMap":
                openweathermap_api_key = api_key
                
                if openweathermap_api_key:
                    st.success("Key Added! The key's validity will be confirmed upon first use.") 
                    os.environ["OPENWEATHERMAP_API_KEY"] = openweathermap_api_key

                else:
                   st.warning("The API key you entered was empty") 
          
        else:
            st.warning("Please enter the API Key for the selected tool.")

    if model_selection==False:
        st.warning("Please select a model.") 

    # Toggle for showing the feedback form
    if st.button("Share Your Feedback"):
        st.session_state.show_form = not st.session_state.get('show_form', False)
        

if not st.session_state.get('show_form', False):
    # Ensure that tool_selection, model_selection, and api_key are defined and valid before calling
    if tool_selection!="" and model_selection!="" and api_key!="":
        main_weather_logic(tool_selection, model_selection)
        
    else:
        st.warning("Please complete the configuration in the sidebar.")

else:
    form_url = "https://forms.gle/rAtVYykA6yWgJz8L6"
    st.markdown(f'<iframe src="{form_url}" width="700" height="520" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>', unsafe_allow_html=True)
        
    # Use on_click callback to change the state
    def toggle_form():
        st.session_state.show_form = False

    # Assign the callback to the button
    if st.button("Return to Weather Information", on_click=toggle_form):
        pass