# Creating a RAG Chatbot With GPT-4o Using SERP Data

[![Promo](https://github.com/luminati-io/LinkedIn-Scraper/raw/main/Proxies%20and%20scrapers%20GitHub%20bonus%20banner.png)](https://brightdata.com/) 

This guide explain how to build a Python RAG chatbot using GPT-4o and Bright Data’s SERP API for more accurate, context-rich AI responses.

1. [Introduction](#how-to-creating-a-rag-chatbot-with-gpt-4o-using-serp-data)
2. [What Is RAG?](#what-is-rag)
3. [Why Feed AI Models With SERP Data](#why-feed-ai-models-with-serp-data)
4. [RAG With SERP Data With GPT Models Using Python: Step-By-Step Tutorial](#rag-with-serp-data-with-gpt-models-using-python-step-by-step-tutorial)
    1. [Step #1: Initialize a Python Project](#step-1-initialize-a-python-project)
    2. [Step #2: Install the Required Libraries](#step-2-install-the-required-libraries)
    3. [Step #3: Prepare Your Project](#step-3-prepare-your-project)
    4. [Step #4: Configure SERP API](#step-4-configure-serp-api)
    5. [Step #5: Implement the SERP Scraping Logic](#step-5-implement-the-serp-scraping-logic)
    6. [Step #6: Extract Text from the SERP URLs](#step-6-extract-text-from-the-serp-urls)
    7. [Step #7: Generate the RAG Prompt](#step-7-generate-the-rag-prompt)
    8. [Step #8: Perform the GPT Request](#step-8-perform-the-gpt-request)
    9. [Step #9: Create the Application UI](#step-9-create-the-application-ui)
    10. [Step #10: Put It All Together](#step-10-put-it-all-together)
    11. [Step #11: Test the Application](#step-11-test-the-application)
5. [Conclusion](#conclusion)

## What Is RAG?

RAG, short for [Retrieval-Augmented Generation](https://blogs.nvidia.comhttps://brightdata.com/blog/what-is-retrieval-augmented-generation/), is an AI approach that combines information retrieval with text generation. In a RAG workflow, the application first retrieves relevant data from external sources—such as documents, web pages, or databases. Then, it passes data to the AI models so that it can generate more contextually relevant responses.

RAG enhances large language models (LLMs) like GPT by enabling them to access and reference up-to-date information beyond their original training data. This approach is key in scenarios where precise and context-specific information is needed, as it improves both the quality and accuracy of AI-generated responses.

## Why Feed AI Models With SERP Data

The knowledge cutoff date for GPT-4o is [October 2023](https://computercity.com/artificial-intelligence/knowledge-cutoff-dates-llms), meaning it lacks access to events or information that came out after that time. However, [GPT-4o models](https://openai.com/index/hello-gpt-4o/) can pull in data from the Internet in real-time using Bing search integration. That helps them offer more up-to-date information and responses that are detailed, precise, and contextually rich.

## RAG With SERP Data With GPT Models Using Python: Step-By-Step Tutorial

This tutorial guides through building a RAG chatbot using OpenAI’s GPT models. The idea is to gather text from the top-performing pages on Google for a specific search query and use it as the context for a GPT request.

The biggest challenge is scraping SERP data. Most search engines come with advanced anti-bot solutions to prevent automated access to their pages. For detailed guidance, refer to our guide on [how to scrape Google in Python](https://brightdata.com/blog/web-data/scraping-google-with-python).

To simplify the scraping process, we will use [Bright Data’s SERP API](https://brightdata.com/products/serp-api).

This SERP scraper allows you to easily retrieve SERPs from Google, DuckDuckGo, Bing, Yandex, Baidu, and other search engines using simple HTTP requests.

We will then extract text data from the returned URLs using a [headless browser](https://brightdata.com/blog/web-data/best-headless-browsers). Then, we will use that information as the context for the GPT model in a RAG workflow. If you instead want to retrieve online data directly using AI, read our article on [web scraping with ChatGPT](https://brightdata.com/blog/web-data/web-scraping-with-chatgpt).

All the code in this guide is also available in a GitHub repository:

```bash
git clone https://github.com/Tonel/rag_gpt_serp_scraping
```

Follow the instructions in the README.md file to install the project’s dependencies and launch the project.

Keep in mind that the approach presented in this blog post can easily be adapted to any other search engine or LLM.

> **Note**:\
> This guide refers to Unix and macOS. If you are a Windows user, you can still follow the tutorial by using the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

### Step #1: Initialize a Python Project

Make sure you have Python 3 installed on your machine. Otherwise, [download and install it](https://www.python.org/downloads/).

Create a folder for your project and switch to it in the terminal:

```bash
mkdir rag_gpt_serp_scraping

cd rag_gpt_serp_scraping
```

The `rag_gpt_serp_scraping` folder will contain your Python RAG project.

Then, load the project directory in your favorite Python IDE. [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) or [Visual Studio Code with the Python extension](https://code.visualstudio.com/docs/languages/python) will do.

Inside rag\_gpt\_serp\_scraping, add an empty app.py file. This will contain your scraping and RAG logic.

Next, initialize a [Python virtual environment](https://docs.python.org/3/library/venv.html) in the project directory:

```bash
python3 -m venv env
```

Activate the virtual environment with the command below:

```bash
source ./env/bin/activate
```

### Step #2: Install the Required Libraries

This Python RAG project will be using the following dependencies:

*   [`python-dotenv`](https://pypi.org/project/python-dotenv/): It will be used to securely manage sensitive credentials, such as Bright Data credentials and OpenAI API keys.
*   [`requests`](https://pypi.org/project/requests/): To perform HTTP requests to Bright Data’s SERP API.
*   [`langchain-community`](https://pypi.org/project/langchain-community/): It will be used for retrieving text from the Google SERP pages and cleaning it to generate relevant content for RAG.
*   [`openai`](https://pypi.org/project/openai/): It will be employed to interface with GPT models to generate natural language responses based on the given inputs and RAG context.
*   [`streamlit`](https://pypi.org/project/streamlit/): It will come in handy for creating a UI where users can input their Google search queries and AI prompt, and view the results dynamically.

Install all the dependencies:

```bash
pip install python-dotenv requests langchain-community openai streamlit
```

We will use [AsyncChromiumLoader](https://python.langchain.com/docs/integrations/document_loaders/async_chromium/) from langchain-community, which requires the following dependencies:

```bash
pip install --upgrade --quiet playwright beautifulsoup4 html2text
```

To function properly, Playwright also requires you to install the browsers:

```bash
playwright install
```

### Step #3: Prepare Your Project

In `app.py`, add the following imports:

```python
from dotenv import load_dotenv

import os

import requests

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI

import streamlit as st
```

Then, create a `.env` file in your project folder to store all your credentials. Your project structure will now look like as below:

![Project structure](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-19.png)

Use the function below in `app.py` to instruct `python-dotenv` to load the environment variables from `.env`:

```python
load_dotenv()
```

You can now import environment variables from `.env` or the system with:

```python
os.environ.get("<ENV_NAME>")
```

### Step #4: Configure SERP API

We will use Bright Data’s SERP API to retrieve content from search engine results pages and use that in our Python RAG workflow. Specifically, we will extract text from the URLs of the web pages returned by the SERP API.

To set up SERP API, refer to the [official documentation](https://docs.brightdata.com/scraping-automation/serp-api/quickstart). Alternatively, follow the instructions below.

If you have not already created an account, [sign up for Bright Data](https://brightdata.com). Once logged in, navigate to your account dashboard:

![Account main dashboard](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-18.png)

There, click the “Get proxy products” button.

That will bring you to the page below, where you have to click on the “SERP API” row:

![Clicking on SERP API](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-17.png)

On the SERP API product page, toggle “Activate zone” to enable the product:

![Activating the SERP zone](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-16.png)

Now, copy the SERP API host, port, username, and password in the “Access parameters” section and add them to your `.env` file:

```python
BRIGHT_DATA_SERP_API_HOST="<YOUR_HOST>"

BRIGHT_DATA_SERP_API_PORT=<YOUR_PORT>

BRIGHT_DATA_SERP_API_USERNAME="<YOUR_USERNAME>"

BRIGHT_DATA_SERP_API_PASSWORD="<YOUR_PASSWORD>"
```

Replace the `<YOUR_XXXX>` placeholders with the values provided by Bright Data on the SERP API page.

Note that the host in “Access parameters” has a format like this:

```python
brd.superproxy.io:33335
```

Split it as below:

```python
BRIGHT_DATA_SERP_API_HOST="brd.superproxy.io"

BRIGHT_DATA_SERP_API_PORT=33335
```

### Step #5: Implement the SERP Scraping Logic

In `app.py`, add the following function to retrieve the first `number_of_urls` URLs from a Google SERP page:

```python
def get_google_serp_urls(query, number_of_urls=5):

# perform a Bright Data's SERP API request

# with JSON autoparsing

host = os.environ.get("BRIGHT_DATA_SERP_API_HOST")

port = os.environ.get("BRIGHT_DATA_SERP_API_PORT")

username = os.environ.get("BRIGHT_DATA_SERP_API_USERNAME")

password = os.environ.get("BRIGHT_DATA_SERP_API_PASSWORD")

proxy_url = f"http://{username}:{password}@{host}:{port}"

proxies = {"http": proxy_url, "https": proxy_url}

url = f"https://www.google.com/search?q={query}&brd_json=1"

response = requests.get(url, proxies=proxies, verify=False)

# retrieve the parsed JSON response

response_data = response.json()

# extract a "number_of_urls" number of

# Google SERP URLs from the response

google_serp_urls = []

if "organic" in response_data:

for item in response_data["organic"]:

if "link" in item:

google_serp_urls.append(item["link"])

return google_serp_urls[:number_of_urls]
```

This makes an HTTP GET request to SERP API with the search query specified in the query argument. The [`brd_json=1`](https://docs.brightdata.com/scraping-automation/serp-api/parsing-search-results) query parameter ensures that SERP API parses the results into JSON for you, in the format below:

```json
{

"general": {

"search_engine": "google",

"results_cnt": 1980000000,

"search_time": 0.57,

"language": "en",

"mobile": false,

"basic_view": false,

"search_type": "text",

"page_title": "pizza - Google Search",

"code_version": "1.90",

"timestamp": "2023-06-30T08:58:41.786Z"

},

"input": {

"original_url": "https://www.google.com/search?q=pizza&brd_json=1",

"user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12) AppleWebKit/608.2.11 (KHTML, like Gecko) Version/13.0.3 Safari/608.2.11",

"request_id": "hl_1a1be908_i00lwqqxt1"

},

"organic": [

{

"link": "https://www.pizzahut.com/",

"display_link": "https://www.pizzahut.com",

"title": "Pizza Hut | Delivery & Carryout - No One OutPizzas The Hut!",

"image": "omitted for brevity...",

"image_alt": "pizza from www.pizzahut.com",

"image_base64": "omitted for brevity...",

"rank": 1,

"global_rank": 1

},

{

"link": "https://www.dominos.com/en/",

"display_link": "https://www.dominos.com › ...",

"title": "Domino's: Pizza Delivery & Carryout, Pasta, Chicken & More",

"description": "Order pizza, pasta, sandwiches & more online for carryout or delivery from Domino's. View menu, find locations, track orders. Sign up for Domino's email ...",

"image": "omitted for brevity...",

"image_alt": "pizza from www.dominos.com",

"image_base64": "omitted for brevity...",

"rank": 2,

"global_rank": 3

},

// omitted for brevity...

],

// omitted for brevity...

}
```

The last few lines of the function retrieve each SERP URL from the resulting JSON data, select only the first `number_of_urls` URLs, and return them in a list.

### Step #6: Extract Text from the SERP URLs

Define a function that extracts text from each of the SERP URLs:

```python
# Note: Some websites may have dynamic content or anti-scraping measures that could prevent text extraction.
# In such cases, please consider using additional tools like Selenium
def extract_text_from_urls(urls, number_of_words=600): 

# instruct a headless Chrome instance to visit the provided URLs

# with the specified user-agent

loader = AsyncChromiumLoader(

urls,

user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",

)

html_documents = loader.load()

# process the extracted HTML documents to extract text from them

bs_transformer = BeautifulSoupTransformer()

docs_transformed = bs_transformer.transform_documents(

html_documents,

tags_to_extract=["p", "em", "li", "strong", "h1", "h2"],

unwanted_tags=["a"],

remove_comments=True,

)

# make sure each HTML text document contains only a number

# number_of_words words

extracted_text_list = []

for doc_transformed in docs_transformed:

# split the text into words and join the first number_of_words

words = doc_transformed.page_content.split()[:number_of_words]

extracted_text = " ".join(words)

# ignore empty text documents

if len(extracted_text) != 0:

extracted_text_list.append(extracted_text)

return extracted_text_list
```

This function:

1.  Loads web pages from the URLs passed as an argument using a headless Chrome browser instance.
2.  Utilizes [BeautifulSoupTransformer](https://python.langchain.com/v0.2/api_reference/community/document_transformers/langchain_community.document_transformers.beautiful_soup_transformer.BeautifulSoupTransformer.html) to process the HTML of each page and extract text from specific tags (like `<p>`, `<h1>`, `<strong>`, etc.), omitting unwanted tags (like `<a>`) and comments.
3.  Limits the extracted text for each webpage to a number of words specified by the `number_of_words` argument.
4.  Returns a list of the extracted text from each URL.

While the `["p", "em", "li", "strong", "h1", "h2"]` tags are enough to extract text from most web pages, in some specific scenarios, you may need to customize this list of HTML tags. Also, you might have to increase or decrease the target number of words for each text item.

For example, consider the [web page below](https://athomeinhollywood.com/2024/09/19/transformers-one-review/):

![Transformers one review page](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-15.png)

Applying that function to that page will result in this text array:

```python
["Lisa Johnson Mandell’s Transformers One review reveals the heretofore inconceivable: It’s one of the best animated films of the year! I never thought I’d see myself write this about a Transformers movie, but Transformers One is actually an exceptional film! ..."]
```

The list of text items returned by `extract_text_from_urls()` represents the RAG context to feed to the OpenAI model.

### Step #7: Generate the RAG Prompt

Define a function that transforms the AI prompt request and text context into the final RAG prompt:

```python
def get_openai_prompt(request, text_context=[]):

# default prompt

prompt = request

# add the context to the prompt, if present

if len(text_context) != 0:

context_string = "\n\n--------\n\n".join(text_context)

prompt = f"Answer the request using only the context below.\n\nContext:\n{context_string}\n\nRequest: {request}"

return prompt
```

Prompts returned by the previous function when a RAG context is specified have this format:

```
Answer the request using only the context below.

Context:

Bla bla bla...

--------

Bla bla bla...

--------

Bla bla bla...

Request: <YOUR_REQUEST>
```

### Step #8: Perform the GPT Request

First, initialize the OpenAI client at the top of the `app.py` file:

```python
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

This relies on the `OPENAI_API_KEY` environment variable, which you can define directly in your system’s environments or in the `.env` file:

`OPENAI_API_KEY="<YOUR_API_KEY>"`

Replace `<YOUR_API_KEY>` with the value of your [OpenAI API key](https://platform.openai.com/api-keys). If you do not know how to get one, follow the [official guide](https://platform.openai.com/docs/quickstart).

Next, write a function that uses the OpenAI official client to perform a request to the [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) AI model:

```python
def interrogate_openai(prompt, max_tokens=800):

# interrogate the OpenAI model with the given prompt

response = openai_client.chat.completions.create(

model="gpt-4o-mini",

messages=[{"role": "user", "content": prompt}],

max_tokens=max_tokens,

)

return response.choices[0].message.content
```

> **Note**:\
> You can configure any other GPT model supported by the OpenAI API.

If called with a prompt returned by `get_openai_prompt()` that includes a specified text context, `interrogate_openai()` will successfully perform retrieval-augmented generation as intended.

### Step #9: Create the Application UI

Use Streamlit to define a simple [form UI](https://docs.streamlit.io/develop/concepts/architecture/forms) where users can specify:

1.  The Google search query to pass to the SERP API
2.  The AI prompt to send to GPT-4o mini

To do that, use this code:

```python
with st.form("prompt_form"):

# initialize the output results

result = ""

final_prompt = ""

# textarea for user to input their Google search query

google_search_query = st.text_area("Google Search:", None)

# textarea for user to input their AI prompt

request = st.text_area("AI Prompt:", None)

# button to submit the form

submitted = st.form_submit_button("Send")

# if the form is submitted

if submitted:

# retrieve the Google SERP URLs from the given search query

google_serp_urls = get_google_serp_urls(google_search_query)

# extract the text from the respective HTML pages

extracted_text_list = extract_text_from_urls(google_serp_urls)

# generate the AI prompt using the extracted text as context

final_prompt = get_openai_prompt(request, extracted_text_list)

# interrogate an OpenAI model with the generated prompt

result = interrogate_openai(final_prompt)

# dropdown containing the generated prompt

final_prompt_expander = st.expander("AI Final Prompt:")

final_prompt_expander.write(final_prompt)

# write the result from the OpenAI model

st.write(result)
```

The Python RAG script is ready.

### Step #10: Put It All Together

Your `app.py` file should contain the following code:

```python
from dotenv import load_dotenv

import os

import requests

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI

import streamlit as st

# load the environment variables from the .env file

load_dotenv()

# initialize the OpenAI API client with your API key

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_google_serp_urls(query, number_of_urls=5):

# perform a Bright Data's SERP API request

# with JSON autoparsing

host = os.environ.get("BRIGHT_DATA_SERP_API_HOST")

port = os.environ.get("BRIGHT_DATA_SERP_API_PORT")

username = os.environ.get("BRIGHT_DATA_SERP_API_USERNAME")

password = os.environ.get("BRIGHT_DATA_SERP_API_PASSWORD")

proxy_url = f"http://{username}:{password}@{host}:{port}"

proxies = {"http": proxy_url, "https": proxy_url}

url = f"https://www.google.com/search?q={query}&brd_json=1"

response = requests.get(url, proxies=proxies, verify=False)

# retrieve the parsed JSON response

response_data = response.json()

# extract a "number_of_urls" number of

# Google SERP URLs from the response

google_serp_urls = []

if "organic" in response_data:

for item in response_data["organic"]:

if "link" in item:

google_serp_urls.append(item["link"])

return google_serp_urls[:number_of_urls]

def extract_text_from_urls(urls, number_of_words=600):

# instruct a headless Chrome instance to visit the provided URLs

# with the specified user-agent

loader = AsyncChromiumLoader(

urls,

user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",

)

html_documents = loader.load()

# process the extracted HTML documents to extract text from them

bs_transformer = BeautifulSoupTransformer()

docs_transformed = bs_transformer.transform_documents(

html_documents,

tags_to_extract=["p", "em", "li", "strong", "h1", "h2"],

unwanted_tags=["a"],

remove_comments=True,

)

# make sure each HTML text document contains only a number

# number_of_words words

extracted_text_list = []

for doc_transformed in docs_transformed:

# split the text into words and join the first number_of_words

words = doc_transformed.page_content.split()[:number_of_words]

extracted_text = " ".join(words)

# ignore empty text documents

if len(extracted_text) != 0:

extracted_text_list.append(extracted_text)

return extracted_text_list

def get_openai_prompt(request, text_context=[]):

# default prompt

prompt = request

# add the context to the prompt, if present

if len(text_context) != 0:

context_string = "\n\n--------\n\n".join(text_context)

prompt = f"Answer the request using only the context below.\n\nContext:\n{context_string}\n\nRequest: {request}"

return prompt

def interrogate_openai(prompt, max_tokens=800):

# interrogate the OpenAI model with the given prompt

response = openai_client.chat.completions.create(

model="gpt-4o-mini",

messages=[{"role": "user", "content": prompt}],

max_tokens=max_tokens,

)

return response.choices[0].message.content

# create a form in the Streamlit app for user input

with st.form("prompt_form"):

# initialize the output results

result = ""

final_prompt = ""

# textarea for user to input their Google search query

google_search_query = st.text_area("Google Search:", None)

# textarea for user to input their AI prompt

request = st.text_area("AI Prompt:", None)

# button to submit the form

submitted = st.form_submit_button("Send")

# if the form is submitted

if submitted:

# retrieve the Google SERP URLs from the given search query

google_serp_urls = get_google_serp_urls(google_search_query)

# extract the text from the respective HTML pages

extracted_text_list = extract_text_from_urls(google_serp_urls)

# generate the AI prompt using the extracted text as context

final_prompt = get_openai_prompt(request, extracted_text_list)

# interrogate an OpenAI model with the generated prompt

result = interrogate_openai(final_prompt)

# dropdown containing the generated prompt

final_prompt_expander = st.expander("AI Final Prompt")

final_prompt_expander.write(final_prompt)

# write the result from the OpenAI model

st.write(result)
```

### Step #11: Test the Application

Launch your Python RAG application with:

```bash
# Note: Streamlit is designed for lightweight applications. For production-grade deployments, consider using frameworks like Flask or FastAPI.
streamlit run app.py
```
In the terminal, you should see the following output:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501

Network URL: http://172.27.134.248:8501
```

Follow the instructions, and visit `http://localhost:8501` in the browser. Below is what you should be seeing:

![Streamlit app screenshot](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-14.png)

Test the application by using a Google search query as below:

```
Transformers One review
```

And an AI prompt as follows:

```
Write a review for the movie Transformers One
```

Click “Send” and wait while your application processes the request. After a few seconds, you should get a result like this:

![App result screenshot](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-13.png)

If you expand the “AI Final Prompt” dropdown, you will see the complete prompt used by the application for RAG.

## Conclusion

The major challenge with using a Python RAG chatbot is scraping search engines like Google:

1. They frequently alter the structure of their SERP pages.
2. They are protected by some of the most sophisticated anti-bot measures available.
3. Retrieving large volumes of SERP data concurrently is complex and can be expensive.

[Bright Data’s SERP API](https://brightdata.com/products/serp-api) helps you retrieve real-time SERP data from all major search engines with no effort. It also supports RAG and many other applications. Get your free trial now!
