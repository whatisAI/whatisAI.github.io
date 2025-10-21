---
layout: post
title: Beginner’s Guide to Agentic Flows - A Tiny AI Travel Agent
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

*Keywords: LLM, AI Agents, Agentic AI*

As we all embrace the new paradigm of agentics architectures, it helps to walk through the process of building a specialized AI agent to get a deeper understanding. The magic happens because AI agents can leverage your inhouse data as well as leveraging the reasoning ability of LLMs. 

### Who this might interest

This post is for those starting in the AI agentic journey, not those familiar with building agents already. It is intended for people who hear the word "agent" and would like to have a mental map of what is happening where, and a high level understanding of how. Whether agents, workflows and tools are the right choice for a specific product within a business, depends on many factors, including, for example, data sensitivity, latency, tool/API reliability & coverage, cost to serve at scale, maturity of your evaluation and fallback strategy. Having a mental model of agentic flows helps in progressing the conversations and making informed decisions. 


# Case study - Building Flai: a tiny AI travel agent

To make the conversation simple, I will be building a (tiny) travel agent. 

The agent is deployed [here](https://huggingface.co/spaces/cvclpl/small_travel_agent) ( although it is password protected).

Watch the demo here

<figure class="video">
  <video
    controls
    preload="metadata"
    playsinline
    style="max-width:100%;height:auto;"
    src="{{ site.baseurl }}/images/output2.mp4"
    poster="{{ site.baseurl }}/images/output2.mp4">
    <track kind="captions" srclang="en" label="English"
           src="{{ site.baseurl }}/images/output2.mp4">
    <p>Your browser doesn’t support HTML5 video.
       <a href="{{ site.baseurl }}/images/output2.mp4">Download the MP4</a>.</p>
  </video>
  <figcaption>demo.</figcaption>
</figure>

The goal is for users to come to our AI travel agent, Flai (as in Fly with AI, but pronounced the same as the insect). Flai will help the user discover which places to go to. [OpenAI shared a neat demo of a travel agent in July 2025](https://www.youtube.com/watch?v=1jn_RpbPbEc), where the agent has the capacity to interact with the webpage. Unlike the OpenAI demo, this is a very stripped down version that has no GUI interaction with any website, and only has access to data on disk.


### Scope for Flai:
- Real agents and tools, limited data
- Flai only has information about 20 cities in Europe. 
- There are no real time APIs being called. 
- Weather data for each destination and month, and average flight time between destinations was scraped and provided via a csv file. 
- Prices are completely random numbers, for weekday trips lasting 5 days starting on Monday, and 3 - day weekend trips starting on Fridays, from mid October to end of December in a csv file. 


Having a conversation with Flai: step by step

Let’s break down what happens when I interact with Flai our tiny travel agent. I can ask

<br>
<div style="text-align: center;">
<img src="{{ site.baseurl }}/images/Flai1.png" alt="conversation beginning" style="width: 300px;"/> </div>
<br> 

The travel agent responds:

<br>
<div style="text-align: center;">
<img src="{{ site.baseurl }}/images/Flai2.png" alt="conversation response" style="width: 300px;"/> </div>
<br> 


As you can see, the agent processed my request and gave me a response using the flight price information in the csv files, which may not be obvious at first sight, but you can check for yourselves with the attached files. But… What is happening under the hood?


### Step 1: Defining the agent (decision policy)
To define an AI agent we will need to use an LLM with structured tool/function calling support. 
For this demo, I am using the OpenAI SDK, but this can be abstracted to any other framework. 

Our agent will receive the user query, and understand the user intent: do they want to find destinations by weather? Does the traveler want to find destinations by flight time? By price? A combination of above? Do they want to know characteristics about a destination? Whilst the agent can understand the traveler's intent, we are assuming the AI agent itself knows nothing about flight prices, hotel prices, accurate weather, etc, and it will need to go and call some tools. In short, the agent will have a decision policy, and choose whether to answer or call some tools, which ones and with what arguments.

The choice of the agent in production applications is a tradeoff of accuracy (better LLMs will understand and parse intents better), latency and cost. Depending on how specialized or broad your agent is, your user queries and intents may lend themselves to smaller, faster llms. 

<br>
<div style="text-align: center;">
<img src="{{ site.baseurl }}/images/FlaiDiagram.png" alt="Flai diagram" style="width: 300px;"/> </div>
<br> 

### Step 2: Defining the tools
In defining the agent we said we assumed the agent LLM has no accurate information on flight/hotel pricing, or weather and we would like to give it the ability to retrieve up to date information. We give this power to the agent via the tools we make available for it to use. In the diagram above, the agent has four tools made available for it to use. 

Tool aversion
In reality, the agent is an LLM, and will have a lot of information from all the data used in pre-training about. However, we do not want the agent to rely on prior knowledge to give information of hotels, flights, prices, etc, but rather to call the tools that will provide up-to-date, reliable price flight, price duration and weather information. When the agent answers from prior knowledge rather than using the tools provided, also known as tool aversion, results won’t be as expected and we will see hallucinations. 


In the example above, when the user asked “I want to go somewhere for Christmas but don’t want to spend more than 350 on return flights”, the agent called the tool “get_desintations_with_flight_price_limit”

Since the tool only  has flight prices from October to December, if it is asked about flight prices outside this range the agent will say it does not have that information. 



```
instructions = (
   "You are a helpful travel assistant that will provide destination recommendations. "
   "When the user asks about prices, dates, temperatures, months, rainfall, durations, or origins/destinations, "
   "you MUST call the relevant tools and ground your answer in those results. "
   "Do not talk about anything that is not travel related or inappropriate and take the user back to travel recommendations."
)
agent = Agent(
   name="Flai: Tiny Travel Agent",
   instructions=instructions,
   tools=[
       get_cities_with_weather_conditions,
       get_citites_within_max_time_flight,
       get_destinations_with_flight_price_limit,
   ],
)
```

### Step 3: Agent loop (Runner / Orchestrator)
Once the tools are called and the tool outputs have been received, the orchestrator has to make a decision whether to call another tool, answer, stop, etc. 

The orchestrator will also be handling retries, time outs and logging requests and tool calling. 
For example, if there are no flight results for a certain query the orchestrator will decide how to relax some constraints. 



As discussed, the agent is calling tools that make use of the datasets I have uploaded. 
Although the evaluation framework is not in place, I have validated a handful of prompts and the flight prices provided by the demo are not made up but rather taken from the dummy data in the csv files available in the tools. As mentioned above, when the agent is asked for flight prices outside the October-December range it will correctly say it does not have that data. 


Note how in this demo there is no tool specific for destination descriptions or activities. Since the agent finds no tools for destination descriptions, the information is coming from the pretrained LLM. Allowing this for a commercial product is not recommended, unless the LLM responses have been evaluated for the desired criteria (factuality,  appropriateness, tone, etc)


### Traces: tool calls throughout the demo conversation
Going back to the demo, we can see in the logs what is happening at each user request. 

For my first request, *"I want to go somwehere for Christmas for under 350 euros"*, I can see in the logs that the first tool that is called is:

```
get_destinations_with_flight_price_limit({
  "origin_city": "London",
  "max_price": 350,
  "min_price": 0
})
```

I got a list of destination recommendations, and asked for those destinations, which one had the highest temperature. At that point, the following tool call happens

```
get_cities_with_weather_conditions({
  "list_months": [
    "December"
  ],
  "min_temp": null,
  "max_temp": null,
  "max_avg_precipitation": null
})
```

Notice that the tool will return the temepratures for the month of December, for all cities. It is the reasoning abilities in the LLM that allow it to focus on the destinations that fit my budget from the previous interaction. 


I then asked, *"If I go to Istanbul, which are the cheapest dates to fly"*, for which the following tool call happens:

```
get_flight_prices_origin_to_destination({
  "destination_city": "Istanbul",
  "origin_city": "London"
})
```

Once again, the tool returns flight prices and the reasoning of the cheapest is done thanks to the reasoning abilities of the LLM.

## Going beyond the small demo

Besides the obvious aspects of scaling the demo to more than 20 destinations and having API calls instead of loading dummy data from memory, there are other areas that could be considered as agents are scaled

### Scoring and Explanation
With a bigger list of destinations and flight price data we are likely to get more options to choose from. It would be beneficial to send these candidates through a ranking model to provide some order based on other factors such as previous user behaviour, popularity, diversity, etc
A few options could be accompanied by an explanation in line with the user’s trip requirements.

### Evaluation
Create a golden set of prompts and do some automatic checks on whether the right tools were called. 
Check if the user preferences were respected. 

### Observability
I have implemented minimal tool tracing for this demo that allowed me to see the tools were being called. However, for production level use cases, we will want to make sure the tools are being called, log the parameters with which they are called, and set up independent evaluations specifically for tool calling. 

### Design
The goal of this small demo was to use a conversational interface, but we can of course imagine the inputs from the user to be coming from a mixture of conversational and interactive widgets such as filters, radio buttons, sliders, etc

### Memory
Ideally, we want to be able to give the ability to the agent to have information to the previous conversations with the same user. 

### Scalability and caching
For a full scale production model with full destination coverage, retrieving all the required information to answer queries could require a lot of API calls that may not be realistic to achieve. 
For example, getting all real time flight price for all destinations within a 6 hour flight might not be achievable, so some work needs to be done in order to achieve this. 

### Fallback strategies
When the tools don’t provide any results, define strategies to relax constraints. 

### Guardrails
Have timeouts on call, retries with backoff, manage unsafe user queries.

### Multi-agent 
There is no reason to stick to a single agent. For example, as in the [October 2025 OpenAI travel agent demo](https://www.youtube.com/watch?v=44eFf-tRiSg) ,  We could have a more modular approach with a flight agent, an itinerary agent, and more, where each one has their own specialization with different tools and different ux components. One of the benefits of having a more modular approach comes when the number of tools, and a lot of context has to be given on the instructions on how or when to use or not those tools. 



