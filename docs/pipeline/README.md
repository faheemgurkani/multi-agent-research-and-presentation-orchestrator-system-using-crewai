# Multi-Agent AI Research & Presentation Generator

## Project Overview

This project implements a **multi-agent AI pipeline** for automating research synthesis, insight analysis, keynote speech writing, and slide design. Built using the **CrewAI** framework and **Gradio** for UI, this system takes a user-defined research topic as input and produces:

* A structured research summary
* Analytical insights grouped by themes
* A polished keynote speech
* A slide-wise presentation outline
* An optional presentation script generated from the analysis and slides

The system leverages state-of-the-art language models (LLMs), function-calling capabilities, web search tools, and agent-based task orchestration.

## Pipeline Architecture

The pipeline is organized into the following components:

### 1. Environment Setup

**Tools Used:**

* `dotenv`
* `os`

**Functionality:**

* Loads API keys and configuration settings from a `.env` file.
* Prepares environment variables for accessing LLM services and search APIs.

### 2. LLM Integration

**Tools Used:**

* `crewai.LLM`
* `mistral-small` model via API

**Functionality:**

* Configures two LLM endpoints:

  * `llm`: used for general response generation
  * `function_calling_llm`: for more structured outputs when necessary
* Parameters such as `temperature` and `max_new_tokens` are tuned for optimal performance.

### 3. Web Search Tool

**Tools Used:**

* `crewai_tools.SerperDevTool`

**Functionality:**

* Provides real-time access to web search results using the Serper API.
* Enhances the research capability of agents by grounding generation in recent information.

### 4. Agent Design

**Tools Used:**

* `crewai.Agent`

**Roles Defined:**

1. **AI Research Specialist**

   * Finds and summarizes recent advances on a given topic.
2. **Insight Analyst**

   * Clusters the research findings into themes and implications.
3. **AI Keynote Writer**

   * Converts the analysis into a compelling 5-minute speech.
4. **Slide Content Designer**

   * Breaks the speech into a slide deck outline.
5. **Presentation Script Writer**

   * Generates a spoken-style presentation script based on the analysis and slides.

### 5. Task Orchestration

**Tools Used:**

* `crewai.Task`
* `crewai.Crew`

**Functionality:**

* Each task is assigned to a corresponding agent.
* Output from one task is passed as input context to the next.
* Tasks generate file outputs to:

  * `results/research_summary.txt`
  * `results/analysis.txt`
  * `results/speech.txt`
  * `results/slides.txt`
  * `results/presentation_script.txt` (optional)

### 6. Gradio Interface

**Tools Used:**

* `gradio.Blocks`

**Functionality:**

* Provides a web-based UI for end-users.
* Accepts research topic input.
* Displays output from all stages using tabbed views:

  * Research Summary
  * Analysis
  * Speech
  * Slide Outline
  * Presentation Script (triggered after generation is complete)

## Technology Stack

| Category           | Tools/Packages                      |
| ------------------ | ----------------------------------- |
| Agent Framework    | `crewai`, `crewai_tools`            |
| LLMs               | `crewai.LLM`, `Mistral API`         |
| Search Integration | `SerperDevTool`                     |
| UI Framework       | `gradio`                            |
| Configuration      | `dotenv`, `os`                      |
| Optional Backends  | `WatsonxLLM`, `HuggingFaceEndpoint` |

## Setup Instructions

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**

   Create a `.env` file in the root directory with the following keys:

   ```env
   SERPER_API_KEY=your_serper_key
   MISTRAL_API_KEY=your_mistral_api_key
   ```

3. **Run the Application**

   ```bash
   python main.py
   ```

   Access the application at `http://127.0.0.1:7860`

## Example Usage

1. Enter a research topic (e.g., "Recent advances in quantum computing")
2. Click "Generate"
3. Navigate between the following output tabs:

   * Research Summary
   * Analysis
   * Keynote Speech
   * Slide Deck Content
   * Presentation Script (click button after pipeline completes)

## Output Format

Each output is saved to the `results/` directory and simultaneously rendered in the Gradio UI.

## Conclusion

This project demonstrates how modular agent-based orchestration combined with modern LLMs can automate high-value knowledge workflows. It is well-suited for academic researchers, keynote speakers, and innovation consultants who need structured insights and compelling narratives generated from raw research topics.
