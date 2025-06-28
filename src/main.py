from crewai import Crew, Task, Agent, LLM
from crewai_tools import SerperDevTool
# from langchain_ibm import WatsonxLLM
from dotenv import load_dotenv
import os
import gradio as gr
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain.llms import HuggingFacePipeline
# from transformers import pipeline



# Loading environment variables
load_dotenv()

# # For, testing
# print("API Key:", os.getenv("WATSONX_APIKEY"))
# print("URL:", os.getenv("WATSONX_URL"))
# print("Model ID:", os.getenv("WATSONX_MODEL_ID"))
# print("Project ID:", os.getenv("WATSONX_PROJECT_ID"))

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["WATSONX_APIKEY"] = os.getenv("WATSONX_APIKEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
os.environ["WATSONX_URL"] = os.getenv("WATSONX_URL")
os.environ["WATSONX_PROJECT_ID"] = os.getenv("WATSONX_PROJECT_ID")

# hf_pipeline = pipeline(
#     "text-generation",
#     model="google/flan-t5-base",    # A bigger would provide with the desired results and behaviour
#     max_new_tokens=512,
#     do_sample=True
# )

# LLM parameters
params = {
    "decoding_method": "greedy",
    "max_new_tokens": 512
}

# # LLM setup
# llm = WatsonxLLM(
#     model_id=os.getenv("WATSONX_MODEL_ID"),
#     url=os.getenv("WATSONX_URL"),
#     params=params,
#     project_id=os.getenv("WATSONX_PROJECT_ID")
# )

llm = LLM(
    model="mistral-small",
    temperature=0.7,
    base_url="https://api.mistral.ai/v1", 
    api_key=os.getenv("MISTRAL_API_KEY") 
)

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct", 
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#     provider="groq",
#     task="auto"
# )

# llm = HuggingFacePipeline(pipeline=hf_pipeline)

# function_calling_llm = WatsonxLLM(
#     model_id=os.getenv("WATSONX_MODEL_ID"),
#     url=os.getenv("WATSONX_URL"),
#     params=params,
#     project_id=os.getenv("WATSONX_PROJECT_ID")
# )

function_calling_llm = LLM(
    model="mistral-small",
    temperature=0.7,
    base_url="https://api.mistral.ai/v1", 
    api_key=os.getenv("MISTRAL_API_KEY") 
)

# function_calling_llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct", 
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#     provider="groq",
#     task="auto"
# )

# function_calling_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# # For, testing
# try:
#     reply = llm.invoke("Explain quantum computing in simple terms.")
#     print("✅ LLM Test Success:\n", reply)
# except Exception as e:
#     print("❌ LLM Test Error:", e)

# Tools
search_tool = SerperDevTool(
    api_key=os.getenv("SERPER_API_KEY"),
)

# Agents
researcher = Agent(
    llm=llm,
    function_calling_llm=function_calling_llm,
    role="AI Research Specialist",
    goal="Identify the latest breakthroughs in the user-defined research topic",
    backstory="A brilliant AI researcher who mines cutting-edge papers, news, and trends to identify valuable insights.",
    tools=[search_tool],
    verbose=1
)

analyst = Agent(
    llm=llm,
    function_calling_llm=function_calling_llm,
    role="Insight Analyst",
    goal="Analyze and cluster the key findings into meaningful themes and extract implications",
    backstory="A domain analyst who translates raw research into structured insights for presentations.",
    verbose=1
)

speech_writer = Agent(
    llm=llm,
    function_calling_llm=function_calling_llm,
    role="AI Keynote Writer",
    goal="Turn findings into an engaging and persuasive speech",
    backstory="A seasoned speechwriter who crafts compelling narratives from research material.",
    verbose=1
)

slide_designer = Agent(
    llm=llm,
    function_calling_llm=function_calling_llm,
    role="Slide Content Designer",
    goal="Generate slide-wise breakdown of the key points for an effective presentation",
    backstory="A visual storytelling expert who distills content into slide-ready formats.",
    verbose=1
)

# Dynamic runner function
def run_multiagent_pipeline(user_topic):
    task_1 = Task(
        description=f"Search and summarize 5 recent advances in: {user_topic}",
        expected_output="A list of 5 key research findings with details.",
        agent=researcher,
        output_file="./results/research_summary.txt"
    )

    task_2 = Task(
        description="Analyze the findings from task 1 and group them into themes with implications",
        context=[task_1],
        expected_output="Structured summary with grouped insights and impact statements",
        agent=analyst,
        output_file="./results/analysis.txt"
    )

    task_3 = Task(
        description="Write a keynote speech based on the analysis in task 2",
        context=[task_2],
        expected_output="A polished and professional 5-minute keynote speech.",
        agent=speech_writer,
        output_file="./results/speech.txt"
    )

    task_4 = Task(
        description="Design a slide deck outline to support the speech from task 3",
        context=[task_3],
        expected_output="Slide-wise content suitable for a presentation.",
        agent=slide_designer,
        output_file="./results/slides.txt"
    )

    crew = Crew(
        agents=[researcher, analyst, speech_writer, slide_designer],
        tasks=[task_1, task_2, task_3, task_4],
        verbose=1
    )

    crew.kickoff()

    # Reading the generated outputs
    def read_file(path):
    
        with open(path, "r", encoding="utf-8") as f:
    
            return f.read()

    return (
        read_file("./results/research_summary.txt"),
        read_file("./results/analysis.txt"),
        read_file("./results/speech.txt"),
        read_file("./results/slides.txt")
    )

# Gradio UI
# gr.Interface(
#     fn=run_multiagent_pipeline,
#     inputs=gr.Textbox(label="Enter your Research Topic or Keywords"),
#     outputs=gr.Textbox(label="Final Output (Speech & Slide Summary)"),
#     title="Multi-Agent AI Research & Presentation Generator"
# ).launch()

with gr.Blocks(title="Multi-Agent AI Research & Presentation Generator") as demo:
    topic_input = gr.Textbox(label="Please input your Research Topic or Keywords")

    run_button = gr.Button("Generate")

    with gr.Tabs():
        with gr.TabItem("🔬 Research Summary"):
            research_output = gr.Textbox(lines=20, label="Research Summary")
        with gr.TabItem("📊 Analysis"):
            analysis_output = gr.Textbox(lines=20, label="Analysis")
        with gr.TabItem("🗣️ Speech"):
            speech_output = gr.Textbox(lines=20, label="Keynote Speech")
        with gr.TabItem("🖼️ Slide Outline"):
            slides_output = gr.Textbox(lines=20, label="Slide Deck Content")

    run_button.click(
        fn=run_multiagent_pipeline,
        inputs=topic_input,
        outputs=[research_output, analysis_output, speech_output, slides_output]
    )

demo.launch()
