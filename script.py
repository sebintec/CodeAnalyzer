import os
import glob
import langchain as lc
import numpy as np
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from datetime import datetime

os.environ["OPENAI_API_KEY"] = 'your-openai-api-key-here'

# Paths and patterns
local_repo_path = 'local-repo-path-here'
file_patterns = ['*.json', '*.txt', '*.py', '*.md']

# Function to get files from local directory
def get_files_from_dir(dir_path, patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(dir_path + '/' + pattern, recursive=True))
    return files

# Function to read content of the files
def read_files(file_paths):
    file_contents = {}
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            file_contents[file_path] = f.read()
    return file_contents

# Function to break a large text into chunks
def get_chunks_from_text(text, num_chunks=10):
    words = text.split()
    words_per_chunk = len(words) // num_chunks
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = ' '.join(words[i:i+words_per_chunk])
        chunks.append(chunk)
    return chunks

# Function to summarize chunks using OpenAI
def summarize_chunks(chunks, prompt_template):
    llm = OpenAI(temperature=0, model_name = 'text-davinci-003')
    llm_chain = LLMChain(llm = llm, prompt = prompt_template)
    summaries = []
    for chunk in chunks:
        chunk_summary = llm_chain.apply([{'text': chunk}])
        summaries.append(chunk_summary)
    return summaries

# Function to calculate similarity matrix
def create_similarity_matrix(chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([' '.join(chunk.split()[:200]) for chunk in chunks])
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

# Get the topics from the similarity matrix
def get_topics(similarity_matrix, num_topics=5):
    distances = 1 - similarity_matrix
    kmeans = KMeans(n_clusters=num_topics).fit(distances)
    clusters = kmeans.labels_
    chunk_topics = np.array([np.where(clusters == i)[0] for i in range(num_topics)])
    return chunk_topics

# Function to parse title and summary results
def parse_title_summary_results(results):
    outputs = []
    for result in results:
        split_result = result.split('|')
        if len(split_result) == 2:
            title = split_result[0].strip()
            summary = split_result[1].strip()
            outputs.append({'title': title, 'summary': summary})
    return outputs

# Function to summarize the stage
def summarize_stage(chunks, topics):
    print(f'Start time: {datetime.now()}')

    # Prompt to get title and summary for each topic
    map_prompt_template = """Write a detailed summary on the structure of the provided content which contains code from selected files from a Github repository, which deploys a chatbot system in Microsoft Azure. Please list all necessary details which can be extrapolated later to specific guidelines how to reverse engineer the repository. I am specifically looking for answers on:
i. the specific steps to deploy this resource? Please list all the files that contain the specific tasks that automate the deployment!
ii. relevant files and code sections that I need to alter in case I want to adjust the overall tool of the repository for my use case. Please list all files and name the code sections.
iii. the detailed steps that need to be performed in order to adjust this repository as a project template for customized deployments."""

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # Define the LLMs
    map_llm = OpenAI(temperature=0, model_name = 'text-davinci-003')
    map_llm_chain = LLMChain(llm = map_llm, prompt = map_prompt)

    summaries = []
    for i in range(len(topics)):
        topic_summaries = []
        for topic in topics[i]:
            map_llm_chain_input = [{'text': chunks[topic]}]
            # Run the input through the LLM chain (works in parallel)
            map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)
            stage_1_outputs = parse_title_summary_results([e['text'] for e in map_llm_chain_results])
            # Split the titles and summaries
            topic_summaries.append(stage_1_outputs[0]['summary'])
        # Concatenate all summaries of a topic
        summaries.append(' '.join(topic_summaries))

    print(f'Stage done time {datetime.now()}')

    return summaries

# Main script
if __name__ == "__main__":
    # Fetch files
    files = get_files_from_dir(local_repo_path, file_patterns)
    file_contents = read_files(files)

    # Iterate over files and process
    for file, content in file_contents.items():
        print(f'Processing {file}...')
        chunks = get_chunks_from_text(content)
        prompt_template = PromptTemplate(template='I need to summarize the following text: {text}', input_variables=['text'])

        # Summarize chunks
        chunk_summaries = summarize_chunks(chunks, prompt_template)

        # Create similarity matrix
        similarity_matrix = create_similarity_matrix(chunks)

        # Get topics
        topics = get_topics(similarity_matrix)

        # Summarize stage
        stage_summary = summarize_stage(chunk_summaries, topics)

        print(f'Summary for {file}:\n{stage_summary}\n')

    print('All files processed.')
