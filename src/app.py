import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import os
import pickle
import base64
#text libraries
import re
from time import time  # To time our operations

from gensim.models import Word2Vec

# used for text scraping
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

def inject(tag):
    with open(os.path.dirname(st.__file__) + "/static/index.html", 'r') as file:
        str = file.read()
        if str.find(tag) == -1:
            idx = str.index('<head>')
            new_str = str[:idx] + tag + str[idx:]
            with open(os.path.dirname(st.__file__) + "/static/index.html", 'w') as file:
                file.write(new_str)

st.session_state['stopwords'] = {'also', 'often', 'may', 'use', 'within', 'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'} 
st.session_state['ignore_stop_words'] = True

def text_to_words(raw_text, remove_stopwords=True):
    # 1. Remove non-letters, but including numbers
    letters_only = re.sub("[^0-9a-zA-Z]", " ", raw_text)
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = st.session_state['stopwords'] # In Python, searching a set is much faster than searching
        meaningful_words = [w for w in words if not w in stops] # Remove stop words
        words = meaningful_words
    return words 

def extract_visible_text(soup):
    result = []
    visible_text = soup.getText()
    sentences = visible_text.splitlines()
    for sentence in sentences:
        words = text_to_words(sentence, remove_stopwords=st.session_state['ignore_stop_words'])
        if len(words) > 5:
            result.append(words)
    return result

def fill_scrape_stats(url, result):
    word_count = 0
    char_count = 0
    for s in result:
        word_count = word_count + len(s)
        for w in s:
            char_count = char_count + len(w)
    if 'url_scrape_stats' in st.session_state:
        print({'Url' : url, 'Characters' : char_count, 'Words' : word_count, 'Sentences' : len(result)})
        st.session_state['url_scrape_stats'].append({'Url' : url, 'Characters' : char_count, 'Words' : word_count, 'Sentences' : len(result)})

def crawler(url, maxurls, pages_crawled):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    result = extract_visible_text(soup)
    fill_scrape_stats(url, result)
    
    # now follow links
    links = soup.find_all('a')

    for link in links:
        if 'href' in link.attrs:
            if not link['href'].startswith('http'):
                link['href'] = "https://" + urlparse(url).netloc.lower() + link['href']

            if link['href'] not in pages_crawled:
                pages_crawled.append(link['href'])
                try:
                    if len(pages_crawled) < maxurls: # stop condition of maximum number of crawled pages
                        print(link['href'])
                        # only follow links within the same domain as the crawler was called for
                        if link['href'].lower().find(urlparse(url).netloc.lower()) != -1: 
                            result.extend(crawler(link['href'], maxurls, pages_crawled))
                except:
                    continue   
    return result

inject('<script type="text/javascript" src="https://js-cdn.dynatrace.com/jstag/148709fdc4b/bf74387hfy/3111aefc2c2c3580_complete.js" crossorigin="anonymous"></script>')    

st.title('Text Scraping and Model Studio')

st.markdown('Data application for scraping Web texts from given Urls for the purpose of training word embedding models. Source at [GitHub](https://github.com/wolfgangB33r/ai-text-model-studio), read the companion [blog](https://www.smartlab.at/web-text-scraping-and-ai-model-training-with-streamlit/).')

st.subheader('Web text scraping')
st.markdown('Enter a Web Url below and start scraping all visual texts from that page.')
max_links = st.slider('Maximum number of links to follow', 0, 100, 1)
scraping_url = st.text_area('Url to scrape texts from (e.g.: texts from the book Pride And Prejudice)', 'https://www.gutenberg.org/cache/epub/1342/pg1342.html')

st.subheader('Text cleaning')
ignore_stop_words = st.checkbox('Ignore stopwords?', value = True)
if ignore_stop_words:
    st.session_state['ignore_stop_words'] = ignore_stop_words

stop_word_text = st.text_area('Stopwords', 'must, much, us, could, would, also, often, may, use, within, ourselves, hers, between, yourself, but, again, there, about, once, during, out, very, having, with, they, own, an, be, some, for, do, its, yours, such, into, of, most, itself, other, off, is, s, am, or, who, as, from, him, each, the, themselves, until, below, are, we, these, your, his, through, don, nor, me, were, her, more, himself, this, down, should, our, their, while, above, both, up, to, ours, had, she, all, no, when, at, any, before, them, same, and, been, have, in, will, on, does, yourselves, then, that, because, what, over, why, so, can, did, not, now, under, he, you, herself, has, just, where, too, only, myself, which, those, i, after, few, whom, t, being, if, theirs, my, against, a, by, doing, it, how, further, was, here, than')
if stop_word_text:
    a = stop_word_text.split(',')
    ca = []
    for i in a:
        ca.append(i.strip())
    st.session_state['stopwords'] = set(ca)

scrapebutton = st.button('Start scraping')
    

if scrapebutton:
    st.session_state['url_scrape_stats'] = []
    urls = scraping_url.splitlines()
    st.session_state['sentences'] = []
    for url in urls:
        st.session_state['sentences'].extend(crawler(url, maxurls = max_links, pages_crawled = []))
        print(len(st.session_state['sentences']))
    if 'url_scrape_stats' in st.session_state:
        df = st.dataframe(st.session_state['url_scrape_stats'])
    base64_str = base64.b64encode(str(st.session_state['sentences']).encode('utf-8')).decode()
    href = f'<a href="data:file/output_model;base64,{base64_str}" download="sentences.txt">Download scraped sentences as JSON file</a>'
    st.markdown(href, unsafe_allow_html=True)
	

st.subheader('Train word2vec')
vector_size = st.slider('Embedding vector size of each word', 0, 500, 100)
window_size = st.slider('Word window size (5 e.g.: means two words before and two words after the input word are taken into account)', 0, 10, 5)

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download trained model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

trainbutton = st.button('Start training')
if trainbutton:
    if 'sentences' in st.session_state:
        # Initialize and train the model (this will take some time)
        model = Word2Vec(vector_size=vector_size, window=window_size, min_count=1, workers=4)
        model.build_vocab(st.session_state['sentences'], progress_per=10000)
        t = time()
        model.train(st.session_state['sentences'], total_examples=model.corpus_count, epochs=1, report_delay=1)
        st.session_state['model'] = model
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        print(len(model.wv))
        #print(model.wv.most_similar(positive=["metric"]))
        #model.save("./model/word2vec.model")
        download_model(model)
        model_info = f'Word2vec model vocabulary size: **{len(model.wv)}**'
        st.markdown(model_info)
    else:
        st.markdown('Start with text scraping first.')

st.subheader('Evaluate the trained model')
st.markdown('Find the top-100 most similar words to the given word below.')
test_word = st.text_input('Check most similar words for', 'sister')
if test_word:
    if 'model' in st.session_state:
        if test_word.lower() in st.session_state['model'].wv:
            similar = st.session_state['model'].wv.most_similar(positive=[test_word.lower()], topn=100)
            st.dataframe(similar)







