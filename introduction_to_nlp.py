import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text_data = """
Searle revised the speech acts classification and claimed that all speech acts fall into five categories: (1) Representative/Assertive: Speech act that expresses speaker’s belief and that commits the speaker to the truth of what is asserted (i.e., words fit the world. Example: 😂 Statements); (2) Directive: Speech act that expresses speaker’s wish and making an attempt to get the hearer to do something (i.e., world fits the words. Example: Requests); (3) Commissive: Speech act that expresses speaker’s intention and marking the commitment for the speaker to engage in future action (I.e., world fits the words. Example: Promise); (4) Expressive: Speech act that expresses speaker’s psychological states which has no direction of fit between the world and words (Example: Apologies) and (5) Declaration: Speech act that brings change in (institutional) reality and has bilateral fit between world and words (Example: Baptizing). 😂
👍 A number of studies have applied speech acts analysis in CMC environments. Vásquez (2011) studied complaints on the travel website TripAdvisor and concluded that complaints co-occurred more frequently with advice and recommendations and they were considered mostly indirect in nature. Other studies focused on users’ self-representation in CMC environments. By examining away messages in Instant Messenger (IM), Nastri et al. (2006) found that they were constructed primarily with assertives, followed by expressives and commissives, but seldom with directives. The authors concluded that away messages tended to reflect both informational and entertainment goals. Similarly, Carr et al. (2012) investigated self-presentation in Facebook status messages and found that they were mostly constructed with expressives, followed by assertives. 😍 Their findings demonstrated differences in how users expressed themselves in alternate media. Given that text-based speech acts often co-occur with emoticons and emojis in CMC, some studies have investigated the relationship between speech acts and emoticon usage in message construction. Dresner and Herring (2010) examined the pragmatic function of emoticons and argued that the primary function of emoticon was not to convey emotion but to indicate an illocutionary force, which is the intended effect of the utterance. While their study provided a more nuanced understanding of the functions of emoticons, their study was not situated in a particular CMC setting. In light of this, Skovholt et al. (2014) investigated the communicative functions of emoticons in workplace emails by adopting speech act theory and politeness theory. Through identification of speech acts followed by emoticons in workplace emails, they found that emoticons contributed to modifying the propositional content and the illocutionary force of speech acts, which corresponded with Drenser and Herring’s results (2010). 😍 More recently, the popularity of emoji use have attracted scholars’ interests. Ge-Stadnyk (2021) examined and compared how social media influencers on Weibo (a Chinese Microblogging site) and Twitter used emoji sequences when engaging in self-presentation. The study identified a variety of text-based speech acts, emoji functions, and functional relations by conducting speech act and pragmatic function analyses and claimed that emoji sequences functioning as ‘emphasis on text’ was most employed in connection with accompanying texts in both Weibo and Twitter data (p. 378). To our best knowledge, studies on speech acts with emoji usage in self-help online discussion forums is sparse. This study expands the current research scope by examining the text-based speech acts and the communicative functions of emoji in an online self-help discussion forum related to COVID-19, with the aim to investigate how Hong Kong forum users framed their COVID-19 experiences, expressed their emotions and seek socioemotional support from others amid a global health crisis. 😍
"""

###**Text normalization**

normalized_text = text_data.lower()
normalized_text

###**Tokenization**

tokens =nltk.word_tokenize(normalized_text)
tokens

###**Stop-word removal**

stop_words = set(stopwords.words('english'))
filtered_tokens = [words for words in tokens if words not in stop_words]
filtered_tokens

###**Stemming**

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens ]
stemmed_tokens

###**Lemmatization**

nlp = spacy.load('en_core_web_sm')
doc = nlp(' '.join(filtered_tokens))
lemmatized_tokens = [token.lemma_ for token in doc]
lemmatized_tokens

###**Text Encoding**

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(lemmatized_tokens)
encoded_labels

###**Vectorization and Embeddings**

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([' '.join(lemmatized_tokens)])
vectors.toarray()

###**Padding/Truncation**

sequences = [encoded_labels]
padded_sequences = pad_sequences(sequences, maxlen = 100, padding = 'post', truncating = 'post')
padded_sequences
