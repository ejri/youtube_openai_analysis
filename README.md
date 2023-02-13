<h1 align="center">
Youtube Analyzer 
</h1>

![Screen shot of the app in Streamlit](/Users/ibrahim/Downloads/youtube_openai_analysis/youtube_analyer.png)

[![Video demo](/Users/ibrahim/Downloads/youtube_openai_analysis/youtube_analyer.png)](https://youtu.be/Vd-_E6m9vm0)


# youtube_openai_analysis
A simple app that uses openAI's gpt-3 to summarize a youtube video, transcribe it, and ask questions about the video. All without having to watch the video. 

- [x] If video has transcript already, pull transcript
- [x] Visualize/play the video in the app. 
- [x] If video doesn't have transcription, use OpenAI's Whisper to transcribe
- [x] Use Embeddings to segment the text making it suitable for a chat application 
- [x] Q&A / chat with the video 
- [x] Log embeddings, chat, transcription into a pandas dataframe
- [x] Build a quick/simple app using streamlit
- [x] Alternative option: run streamlit app in colab
- [ ] Semantic search with embedding
- [ ] Sentiment analysis of the video content
- [ ] Graphs, graphs, and graphs about some of the stuff above.

# Running Locally

1. Clone the repository

```bash
git clone https://github.com/ejri/youtube_openai_analysis
cd youtube_openai_analysis
```
2. Install dependencies

These dependencies are required to install with the requirements.txt file:

``` bash
pip install -r requirements.txt
```

3. Run the Streamlit server

```bash
streamlit run app.py
```

# Running on Google Colab

1. Clone the repository

```
!git clone https://github.com/ejri/youtube_openai_analysis
%cd youtube_openai_analysis
```
2. Install dependencies

These dependencies are required to install with the requirements.txt file:

``` 
!pip install -r requirements.txt
```

3. Set up enviroment on colab:

if not installed already: 
```
!pip install pyngrok==5.2.1
```

Setup streamlit and ngrok
```
!streamlit run /content/youtube_openai_analysis/app.py &>/dev/null&
```

Create an account on ngrok, and paste your authenication token ----
```
!ngrok authtoken ----
```

making sure it's the correct version on the colab servers
```
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip /content/youtube_openai_analysis/ngrok-stable-linux-amd64.zip
```

Setting the server to accept running ngrok
```
get_ipython().system_raw('./ngrok http 8501 &')
```

Runs ngrok as localhost
```
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

4. Run the Streamlit server

```
!streamlit run /content/youtube_openai_analysis/app.py
```

