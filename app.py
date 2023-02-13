import pandas as pd
import numpy as np
import streamlit as st
import whisper
from pytube import YouTube
from streamlit_chat import message
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import os
import sys
from youtube_transcript_api import YouTubeTranscriptApi
import re
from time import time,sleep

def get_video_id_from_video_id_or_url(video_id_or_url):
  # if the video id is longer than 11 characters, then it's a url
  if len(video_id_or_url) > 11:
      # if it's a url, cut it into a video id
      return video_id_or_url[-11:]
  else:
      # it's a video id
      return video_id_or_url

def get_chunks_from_youtube(video_id):
    # fetch the transcript of the video, and chunk it into 10min intervals
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    chunks = []

    start_timestamp = 0.0
    current_timestamp_mins = 0.0

    current_chunk = []

    for entry in transcript:
        current_timestamp_mins = entry['start'] / 60.0

        # specify the 10 min chunks. this can be changed into less minutes if max_token error pops up. 
        if current_timestamp_mins - start_timestamp > 10:
            # append the chunks into an array
            chunks.append(current_chunk)
            # reset the start timestamp
            start_timestamp = current_timestamp_mins
            # reset the current chunk
            current_chunk = []

        # add the line to the current chunk
        current_chunk.append(entry['text'])

    # add the last chunk
    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    print(f"Found {len(chunks)} chunks")

    return chunks

def summarize_chunk(index, chunk):
    chunk_str = "\n".join(chunk)
    prompt = f"""The following is a section of the transcript of a youtube video. It is section #{index+1}:
    {chunk_str}
    Summarize this section of the transcript."""

    if diagnostics:
        # print each line of the prompt with a leading # so we can see it in the output
        for line in prompt.split('\n'):
            print(f"# {line}")
    openai.api_key = user_secret
    completion = openai.Completion.create(
        engine="text-davinci-003", 
        max_tokens=500, 
        temperature=0.2,
        prompt=prompt,
        frequency_penalty=0
    )

    msg = completion.choices[0].text
    msg = re.sub('\s+', ' ', msg)
    filename = '%s_log.txt' % time()
    with open('response_logs/%s' % filename, 'w') as outfile:
        outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + msg)
    with open('response.txt', 'w') as f:
        f.write(msg)

    if diagnostics:
        print(f"# Response: {msg}")

    return msg

def summarize_the_summaries(summaries):

    summaries_str = ""
    for index, summary in enumerate(summaries):
        summaries_str += f"Summary of chunk {index+1}:\n{summary}\n\n"

    prompt = f"""The following are summaries of a youtube video in 10 minute chunks:"
    {summaries_str}
    Summarize the summaries."""

    if diagnostics:
        # print each line of the prompt with a leading # so we can see it in the output
        for line in prompt.split('\n'):
            print(f"# {line}")
    
    openai.api_key = user_secret
    completion = openai.Completion.create(
        engine="text-davinci-003", 
        max_tokens=500, 
        temperature=0.2,
        prompt=prompt,
        frequency_penalty=0
    )

    overall_msg = completion.choices[0].text
    overall_msg = re.sub('\s+', ' ', overall_msg)
    filename = '%s_log.txt' % time()
    with open('response_logs/%s' % filename, 'w') as outfile:
        outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + overall_msg)
    with open('response.txt', 'w') as f:
        f.write(overall_msg)

    if diagnostics:
        print(f"# Response: {overall_msg}")

    return overall_msg

def summarization_video(youtube_link):
  
  #video_id_or_url = sys.argv[1]
  video_id_or_url =  youtube_link

  # if the video id or url is a url, extract the video id
  video_id = get_video_id_from_video_id_or_url(video_id_or_url)

  if len(sys.argv) > 2:
      for arg in sys.argv[2:]:
          if arg == "--diagnostics":
              global diagnostics
              diagnostics = True

          if arg == "--mentions":
              global include_mentions
              include_mentions = True

  # chunks = get_chunks(transcript_file_name)
  chunks = get_chunks_from_youtube(video_id)

  if len(chunks) == 0:
      print("No chunks found")
      summaries = []
      summary_of_summaries= []
      return summaries, summary_of_summaries
  elif len(chunks) == 1:
      summary = summarize_chunk(0, chunks[0])
      print(f"\nSummary: {summary}")
      summaries = summary
      summary_of_summaries= []
      return summaries, summary_of_summaries

  else:
      # summarize each chunk
      summaries = []
      for index, chunk in enumerate(chunks):
          summary = summarize_chunk(index, chunk)
          summaries.append(summary)
          print(f"\nSummary of chunk {index+1}: {summary}")

      # summarize the chunk summaries 
      summary_of_summaries = summarize_the_summaries(summaries)

      print(f"\nSummary of summaries: {summary_of_summaries}")
      return summaries, summary_of_summaries

# whisper
model = whisper.load_model('base')
output = ''
data = []
data_transcription = []
data_summarization = []
embeddings = []
mp4_video = ''
audio_file = ''
diagnostics = 0
include_mentions = 0
summaries = []
summary_of_summaries= []

# Sidebar
with st.sidebar:
    user_secret = st.text_input(label = ":blue[OpenAI API key]",
                                placeholder = "Paste your openAI API key, sk-",
                                type = "password")
    youtube_link = st.text_input(label = ":red[Youtube link]",
                                placeholder = "")
    if youtube_link and user_secret:
        youtube_video = YouTube(youtube_link)
        streams = youtube_video.streams.filter(only_audio=True)
        stream = streams.first()
        if st.button("Start Analysis"):
            if os.path.exists("word_embeddings.csv"):
                os.remove("word_embeddings.csv")
            if os.path.exists("transcription.csv"):
                os.remove("transcription.csv")
            if os.path.exists("summarization.csv"):
                os.remove("summarization.csv")
                
            with st.spinner('Running process...'):
                # Get the video mp4
                mp4_video = stream.download(filename='youtube_video.mp4')
                audio_file = open(mp4_video, 'rb')
                st.write(youtube_video.title)
                st.video(youtube_link) 

                # Whisper
                output = model.transcribe("youtube_video.mp4")
                
                # Transcription
                transcription = {
                    "title": youtube_video.title.strip(),
                    "transcription": output['text']
                }
                data_transcription.append(transcription)
                pd.DataFrame(data_transcription).to_csv('transcription.csv') 

                # Embeddings
                segments = output['segments']
                for segment in segments:
                    openai.api_key = user_secret
                    response = openai.Embedding.create(
                        input= segment["text"].strip(),
                        model="text-embedding-ada-002"
                    )
                    embeddings = response['data'][0]['embedding']
                    meta = {
                        "text": segment["text"].strip(),
                        "start": segment['start'],
                        "end": segment['end'],
                        "embedding": embeddings
                    }
                    data.append(meta)
                pd.DataFrame(data).to_csv('word_embeddings.csv')

                # Summary
                summaries, summary_of_summaries = summarization_video(youtube_link)
                summarization = {
                    "title": youtube_video.title.strip(),
                    "summarizations of video in 10mins chunks": summaries,
                    "overall summary": summary_of_summaries
                }
                data_summarization.append(summarization)
                pd.DataFrame(data_summarization).to_csv('summarization.csv')
                st.success('Completed. Check Tabs for details')

st.title("Youtube Analyzer 🤓 ")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Intro", "Transcription", "Embedding", "Chat with the Video", "Summary"])
with tab1:
    st.markdown('A simple app that uses openAI\'s gpt-3 to summarize a youtube video, transcribe it, and ask questions about the video. All without having to watch the video. ')
    st.markdown("""---""")
    st.write('***What this app does:***')
    st.checkbox('Visualize/play the video in the app.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Transcribe.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Create embeddings from the video transcript.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Chat and ask questions about the video.', value=True, disabled=True, label_visibility="visible")
    st.markdown("""---""")
    st.write('***Progress and features:***')
    st.checkbox('Play the youtube video within app.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('If video has transcript already, pull transcript.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('If video doesn\'t have transcription, use OpenAI\'s Whisper to transcribe.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Use Embeddings to segment the text making it suitable for a chatbot application.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Log embeddings, chat, transcription into a pandas dataframe.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Q&A / chat with the video.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Build a quick/simple app using streamlit.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Alternative option: run streamlit app in colab.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Multi-language integration: non-English videos compatibility.', value=True, disabled=True, label_visibility="visible")
    st.checkbox('Multi-language integration: allow users to ask questions in their languages.', value=True, disabled=True, label_visibility="visible")
    st.markdown("""---""")
    st.write('***Main tools used:***')
    st.write("- OpenAI: Whisper, GPT-3.")
    st.write("- Streamlit")
    st.markdown("""---""")
    st.write('Repo: [Github](https://github.com/ejri/youtube_openai_analysis)')

with tab2: 
    st.header("Transcription:")
    if(os.path.exists("youtube_video.mp4")):
        audio_file = open('youtube_video.mp4', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
    if os.path.exists("transcription.csv"):
        df = pd.read_csv('transcription.csv')
        st.write(df)
with tab3:
    st.header("Embeddings:")
    if os.path.exists("word_embeddings.csv"):
        df = pd.read_csv('word_embeddings.csv')
        st.write(df)
with tab4:
    st.header("Ask me about the video:")
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text():
        input_text = st.text_input("You: ","", key="input")
        return input_text

    user_input = get_text()

    def get_embedding_text(api_key, prompt):
        openai.api_key = user_secret
        response = openai.Embedding.create(
            input= prompt.strip(),
            model="text-embedding-ada-002"
        )
        q_embedding = response['data'][0]['embedding']
        df=pd.read_csv('word_embeddings.csv', index_col=0)
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)

        df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
        returns = []
        
        # Sort by distance with 2 hints
        for i, row in df.sort_values('distances', ascending=True).head(4).iterrows():
            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        return "\n\n###\n\n".join(returns)

    def generate_response(api_key, prompt):
        one_shot_prompt = '''I am Youtube Analyzer, a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.
        Q: What is human life expectancy in the United States?
        A: Human life expectancy in the United States is 78 years.
        Q: '''+prompt+'''
        A: '''
        completions = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = one_shot_prompt,
            max_tokens = 1024,
            n = 1,
            stop=["Q:"],
            temperature=0.5,
        )
        message = completions.choices[0].text
        message = re.sub('\s+', ' ', message)
        filename = '%s_log.txt' % time()
        with open('response_logs/%s' % filename, 'w') as outfile:
            outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + message)
        with open('response.txt', 'w') as f:
            f.write(message)
        
        return message

    if user_input:
        text_embedding = get_embedding_text(user_secret, user_input)
        title = pd.read_csv('transcription.csv')['title']
        string_title = "\n\n###\n\n".join(title)
        user_input_embedding = 'Using this context: "'+string_title+'. '+text_embedding+'", answer the following question. \n'+user_input
        # st.write(user_input_embedding)
        output = generate_response(user_secret, user_input_embedding)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
with tab5:    
    st.header("Video Summary:")
    if os.path.exists("summarization.csv"):
        df = pd.read_csv('summarization.csv')
        st.write(df)
