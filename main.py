import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import LdaModel, HdpModel
from gensim import corpora
import spacy
from flask import Flask, jsonify, request
import json
from dotenv import load_dotenv
import os

# load spacy small models for light the model
nlp = spacy.load("en_core_web_md")
# print("Model loading done!")

def clean(Story):
  # remove stop words
  stop_words = list(get_stop_words('en'))         #About 900 stopwords
  nltk_words = list(stopwords.words('english')) #About 150 stopwords
  stop_words.extend(nltk_words)

  # Text to token
  text_tokens = word_tokenize(Story)
  tokens_without_sw = [word.lower() for word in text_tokens if not word in stop_words] # remove stop words

  # lemmetize story
  lemmatizer = WordNetLemmatizer()
  tokens_without_sw = [lemmatizer.lemmatize(i).strip() for i in tokens_without_sw]

  # clean text
  Story = ' '.join(tokens_without_sw).replace(". .",".").replace(" . . ",". ").replace(" . ",". ")
  Story = re.sub('[^A-Za-z.]+', ' ', Story).replace(". . ",". ").replace(" .",". ").strip()
  Story_sent = [re.sub('[^A-Za-z]+', ' ', i).strip() for i in sent_tokenize(Story) if len(i) > 2]
  return Story,Story_sent

def order_subset_by_coherence(dirichlet_model, bow_corpus, num_topics=10, num_keywords=10):
    """
    Orders topics based on their average coherence across the corpus

    Parameters
    ----------
        dirichlet_model : gensim.models.type_of_model
        bow_corpus : list of lists (contains (id, freq) tuples)
        num_topics : int (default=10)
        num_keywords : int (default=10)

    Returns
    -------
        ordered_topics, ordered_topic_averages: list of lists and list
    """
    if type(dirichlet_model) == gensim.models.ldamodel.LdaModel:
        shown_topics = dirichlet_model.show_topics(num_topics=num_topics, 
                                                   num_words=num_keywords,
                                                   formatted=False)
    elif type(dirichlet_model)  == gensim.models.hdpmodel.HdpModel:
        shown_topics = dirichlet_model.show_topics(num_topics=150, # return all topics
                                                   num_words=num_keywords,
                                                   formatted=False)
    model_topics = [[word[0] for word in topic[1]] for topic in shown_topics]
    topic_corpus = dirichlet_model.__getitem__(bow=bow_corpus, eps=0) # cutoff probability to 0 

    topics_per_response = [response for response in topic_corpus]
    flat_topic_coherences = [item for sublist in topics_per_response for item in sublist]

    significant_topics = list(set([t_c[0] for t_c in flat_topic_coherences])) # those that appear
    topic_averages = [sum([t_c[1] for t_c in flat_topic_coherences if t_c[0] == topic_num]) / len(bow_corpus) \
                      for topic_num in significant_topics]

    topic_indexes_by_avg_coherence = [tup[0] for tup in sorted(enumerate(topic_averages), key=lambda i:i[1])[::-1]]

    significant_topics_by_avg_coherence = [significant_topics[i] for i in topic_indexes_by_avg_coherence]
    ordered_topics = [model_topics[i] for i in significant_topics_by_avg_coherence][:num_topics] # limit for HDP

    ordered_topic_averages = [topic_averages[i] for i in topic_indexes_by_avg_coherence][:num_topics] # limit for HDP
    ordered_topic_averages = [a/sum(ordered_topic_averages) for a in ordered_topic_averages] # normalize HDP values

    return ordered_topics, ordered_topic_averages


def theme_extractor(corpus):
	# embedding story to doc2bow model
	dirichlet_dict = corpora.Dictionary(corpus)
	bow_corpus = [dirichlet_dict.doc2bow(text) for text in corpus]

	# set-up parameters
	num_topics = 10
	num_keywords = 300

	# HdpModel model is update version of LDA model 
	dirichlet_model = HdpModel(corpus=bow_corpus, 
	                           id2word=dirichlet_dict,
	                           chunksize=len(bow_corpus))

	# fit model
	ignore_words = ["he","she"] # set keywords which are not required
	ordered_topics, ordered_topic_averages = \
	    order_subset_by_coherence(dirichlet_model=dirichlet_model,
	                              bow_corpus=bow_corpus, 
	                              num_topics=num_topics,
	                              num_keywords=num_keywords)

	keywords = []
	for i in range(num_topics):
	    # Find the number of indexes to select, which can later be extended if the word has already been selected
	    try:
	      selection_indexes = list(range(int(round(num_keywords * ordered_topic_averages[i]))))
	    except:
	      pass
	    if selection_indexes == [] and len(keywords) < num_keywords: 
	        # Fix potential rounding error by giving this topic one selection
	        selection_indexes = [0]
	              
	    for s_i in selection_indexes:
	      try:
	        if ordered_topics[i][s_i] not in keywords and ordered_topics[i][s_i] not in ignore_words:
	            keywords.append(ordered_topics[i][s_i])
	        else:
	            selection_indexes.append(selection_indexes[-1] + 1)
	      except:
	        pass

	# Fix for if too many were selected
	keywords = keywords[:num_keywords]

	return keywords

def filter_themes(keywords,nlp):
	# set semanticin filters to get more accurate themes and topics 
	doc = nlp(" ".join(keywords))
	final_list_2=[]
	for token in doc:
	        if token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "compound" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NNS" and token.dep_ == "conj" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "nsubj" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "ROOT" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "nmod" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "intj" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "dobj" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "appos" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "PROPN" and token.tag_ == "NNP" and token.dep_ == "ROOT" and spacy.explain(token.pos_) == "proper noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "ADJ" and token.tag_ == "JJ" and token.dep_ == "amod" and spacy.explain(token.pos_) == "adjective":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NNS" and token.dep_ == "nsubj" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "PROPN" and token.tag_ == "NNP" and token.dep_ == "compound" and spacy.explain(token.pos_) == "proper noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "npadvmod" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "ROOT" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "VERB" and token.tag_ == "VBP" and token.dep_ == "dobj" and spacy.explain(token.pos_) == "verb":
	          final_list_2.append(token.text)
	        elif token.pos_ == "ADJ" and token.tag_ == "JJ" and token.dep_ == "dobj" and spacy.explain(token.pos_) == "adjective":
	          final_list_2.append(token.text)
	        elif token.pos_ == "VERB" and token.tag_ == "VBN" and token.dep_ == "amod" and spacy.explain(token.pos_) == "verb":
	          final_list_2.append(token.text)
	        elif token.pos_ == "VERB" and token.tag_ == "VBG" and token.dep_ == "xcomp" and spacy.explain(token.pos_) == "verb":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NNS" and token.dep_ == "npadvmod" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "NOUN" and token.tag_ == "NN" and token.dep_ == "conj" and spacy.explain(token.pos_) == "noun":
	          final_list_2.append(token.text)
	        elif token.pos_ == "ADV" and token.tag_ == "RB" and token.dep_ == "advmod" and spacy.explain(token.pos_) == "adverb":
	          final_list_2.append(token.text)

	# list of common keywords (**remarkably important)
	filet_list=["week","made","note","shut","wearing","obsessive","behind","prepared","asked","need","talking","best","match","considering","rips","matching","stepped","tablet","showing","trick","using","explain","curiosity","charge","satisfying","assigned","overhead","notified",'watch','miyu','jerome','telling','chick','reacts','quietly','crone','story','boring','seek','grab','stare','wooden','ground','issue','toss','creek','watching','enjoy','rica','control','seem','help','adam','physical','weatherby','problem','found','spot','suddenly','release','slow','slightly','angrily','idiot','loss','word','continued','around','wake','center','noted','fall','result','realizing','change','understand','staircase','solid','awoke','accepted','thinking','tablet','charlie','complain','home','make','previous','charge','satisfying','match',"previous","pick","weird","else","endure","if","sincerely","noticing","somehow","stared","yesterday","together","certain","hover","think","tend","seat","feel","leaf","cream","lightly","nerve","trickle","choice","freely","here","nose","flashlight","notch","notice","show","flow","mine","giggle","bathroom","checked","catch","pocket",'familiar','adaptation','content','longer','stop','plea','indiscriminately',"onward",'sickeningly',"focus","consume","strain","shower","ineffective","spite","ease","range","controlled","expected","shriek","hesitation","taste","outbreak","instantly","","side","doubt","fixed","taking","mass","throne","period","known","slump","wood","crack","obviously","forth","series","third","fifth","six","sixth","five","seven","eight","nind","ten","proper","thud","sinew","twist","cradling","instantaneous","seemingly","apart","watched","beautiful","half","snap","accompanied","starting","relaxed","people","extremely","welcoming","important","severe","instinctively","count","friendly","welcome","often","stay","somewhat","mention","knife","today","cheap","however","case","pair","otherwise","somewhat","everywhere","comfortable","exact","glass","feeling","felt","bringing","ignore","part","small","annoying","afterwards","online","admire","talk","sprint","also","sweet","white","only","pretty","sitting","roll","black","loud","bastard","finger","","quiet","complete","short","dirt","purpose","gesture","shed","sigh","jerk","crate","shoulder","somewhere","empty","turn","freshly","mound","shovel","area","hard","soft","frantically","wide","probably","field","rest","knee","minute","decade","heart","volume","soon","feed","open","grinning","close","coming","crushing","seen","crush","face","device","hair","foot","matted","smile","lipstick","looking","upstairs","blade","increasingly","occasionally","knelt","table","look","hold","test","point","progress","lowered","negative","uniformed","chair","anywhere","maybe","rephrase","cross","blink","hypnotic","rough","search","squirmed","probe","neck","artery","tilt","sinking","limb","throat","pulse","chin","tongue","brain","unable","kind","front","mind","opened","afford","second","there","answer","sorry","evening","tapped","shaken","keeping","matter","quickly","pearly","flesh","slowly","convincing","talked","silence","passed","convincing","alternative","response","gate","inside","outside","relentlessly","many","example","common","minor","true","false","completely","apple","mouth","wrenched","know","monday","tuesday","wednesday","thursday","friday","saturday","sunday","dusty","forever","followed","plan","original","manage","putting","handed","take","favorite","forward","worried","awhile","sometimes","inch","putting","picked","guess","almost","refuse","hour","month","still","real","fell","stretch","reach","slowly","laughter","joyfully","happy","definition","displaying","number","repeatedly","really","tired","insanity","aloud","saying","simple","thing","meant","just","tell","closed","lead","wrong","stair","thing","stoplight","grey","right","orangish","door","move","plenty","body","disposed","retreat","find","start","plate","downstairs","steamy","greasy","even","yup","mean","going","confine","outline","closed","brutal","bro","list","date","dreary","standing","venture","rush","ordinary","early","hand","final","beeped","left","spectrum","professor","scout","reality","yellow","kid","colored","reason","power","fate","intense","red","based","basically","class","finally","sense","potential","year","tends","fair","reflection","full","sort","hope","different","live","green","sickening","sence","aura","glow","crash","end","numbered","amount","fired","later","fresh","morning","motif","young","t","clutching","burining","pain","fried","gently","excitedly","chest","fruit","breakfast","shaking","floor","courage","place","bath","safe","hours","terrible","spread","hastily","wire","razor","huge","confused","squirming","spasmed","leave","particularly","filled","insect","distance","screaming","tap","back","slide","always","s","hesitant","properly","bad","chance","doctor","boredom","cat","far","normalcy","care","dad","little","park","undammed","pointing","easy","gouged","charming","duration","mom","data","likely","replaced","away","dark","dishwasher","ready","fake","baby","sparingly","several","street","constantly","vitamin","toy","bored","parent","wall","sure","next","scene","name","game","naturally","night","paper","section","bedroom","kinda","new","corner","skin","name","getting","room","actually","day","ceiling","converting","removal","piece","peeling","papered","sunburn","am","instance","ma","person","material","similar","enough","peel","removing","already","oddly","removed","man","head","much","tenth","licked","indeed","use","wife","world","knowing","wristwatch","work","well","too","ever","syndrome","dog","then","moment","life","old","smiled","instead","long","time","forehead","eventually","wisely","overjoyed","fact","boy","eye","last","exactly","used","first","son","alive","pat","good","boy","girl","girls","boys"]

	# remove keyowrds whose length less then 3 and which are on common word list
	main_list = [i for i in final_list_2 if i not in filet_list and len(i) > 3]
	return main_list


def predict_themes(Story,nlp):
	Story,Story_sent = clean(Story) # story cleaning
	corpus = [i.split() for i in Story_sent]  # text to tokens
	keywords_TT = theme_extractor(corpus) # elicit themes and topics
	thems_topics = filter_themes(keywords_TT,nlp) # filter keywords
	return thems_topics

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# set env for secret key
load_dotenv()

secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

# print(secret_id)
def check_for_secret_id(request_data):    
    try:
        if 'secret_id' not in request_data.keys():
            return False, "Secret Key Not Found."
        
        else:
            if request_data['secret_id'] == secret_id:
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False,message

@app.route('/Topic_theme_prd',methods=['POST'])  #main function
def main():
    params = request.get_json()
    input_query=params["data"]
    Story = input_query[0]['Story']
    key = params['secret_id']

    request_data = {'secret_id' : key}
    secret_id_status,secret_id_message = check_for_secret_id(request_data)
    print ("Secret ID Check: ", secret_id_status,secret_id_message)
    if not secret_id_status:
        return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                        'success':False}) 
    else:
    	prediction= predict_themes(Story,nlp)
    	dictonary = {"themes": prediction}

    return	jsonify(dictonary)

if __name__ == '__main__':
    app.run()   