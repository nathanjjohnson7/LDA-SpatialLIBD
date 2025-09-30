from LDA import LDA
from sklearn.feature_extraction.text import CountVectorizer
import spacy

categories = []
contents = []
with open('archive/bbc-news-data.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.strip().split('\t')
        assert len(line) == 4
        categories.append(line[0].strip())
        contents.append(line[-1].strip())

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def lemmatize_texts(texts):
    lemmatized = []
    for doc in nlp.pipe(texts, batch_size=50):  #batch processing
        #keep lemmas for alphabetic tokens that are not stopwords
        lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop] #get lemmas that are words and not stopwords
        lemmatized.append(" ".join(lemmas))
    return lemmatized

lemmatized_texts = lemmatize_texts(contents)

vectorizer = CountVectorizer(stop_words='english', min_df=2)
X = vectorizer.fit_transform(lemmatized_texts)
print(X.shape) #(2225, 13011) -> (num_documents, num_words)

#bag-of-words
bow = X.toarray()

vocab = vectorizer.get_feature_names_out()

lda = LDA(bow, vocab, num_topics = 5)
lda.complete_loop(max_iters=1000, tol=10) #train

preds = np.argmax(lda.gamma, axis=1)
headers = np.unique(categories)
cats = np.array(categories)

final_matrix = np.zeros((headers.shape[0], lda.k)) #(5,5) -> (num real topics, num latent topics)

for i, head in enumerate(headers):
    indices = np.nonzero(cats == head)
    cat_preds = preds[indices]
    unique_values, counts = np.unique(cat_preds, return_counts=True)
    final_matrix[i][unique_values] = counts
    
result = final_matrix.astype(int)
np.set_printoptions(formatter={'int': '{:3d}'.format})  #4 spaces per number
for i, r in enumerate(result):
    print(r, headers[i])
