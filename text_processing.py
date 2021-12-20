import unicodedata
import re
import nltk
import textacy

def replace_polish_letters(text):
    return unicodedata.normalize('NFKD', text).replace(u'ł', 'l').encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_special_characters(text):
    pattern = r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    return text

def remove_stopwords(text, stopwords):
    tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def lemmatize(text):
    pl = textacy.load_spacy_lang('pl_core_news_lg')
    doc = textacy.make_spacy_doc(text, lang=pl)
    lemma_text_list = [(token.lemma_) for token in doc]
    lemma_text = ' '.join(lemma_text_list)
    return lemma_text
    
def preprocess_text(data, polish_letters=True, lowercase=True, special_characters=True, stopwords=True, lemmatization=True):
    
    stopword_list = nltk.corpus.stopwords.words('polish')
    preprocessed_data = []
    
    for document in data:
        if polish_letters:
           document = replace_polish_letters(document)
            
        if lowercase:
           document = document.lower()
            
        if special_characters:
            special_char_pattern = re.compile(r'([{.(-)!}])')
            document = special_char_pattern.sub(" \\1 ", document)
            document = remove_special_characters(document)
        
        document = re.sub(' +', ' ', document)
        
        if stopwords:
            document = remove_stopwords(document, stopword_list)    
             
        if lemmatization:   
            document = lemmatize(document)
            
        preprocessed_data.append(document)
    
    return preprocessed_data