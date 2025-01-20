import nltk
import random
from nltk.corpus import wordnet
from random import choice, sample

nltk.download('wordnet')


def get_synonyms(word, pos=None):
    """获取单词的同义词列表，可选地指定词性"""
    synonyms = set()
    if pos is None:
        for synset in wordnet.synsets(word):
            lemma_names = [lemma.name().replace("_", " ").replace("-", " ").lower()
                           for lemma in synset.lemmas()]
            synonyms.update(lemma_names)
    else:
        for synset in wordnet.synsets(word, pos=pos):
            lemma_names = [lemma.name().replace("_", " ").replace("-", " ").lower()
                           for lemma in synset.lemmas()]
            synonyms.update(lemma_names)
    return list(synonyms)

def shuffle_sentences(text):
    sentences = text.split('. ')
    random.shuffle(sentences)
    return '. '.join(sentences)

def add_noise(text, noise_rate=0.05):
    noise_words = ["blah", "random", "text", "noise", "example"]
    words = text.split()
    new_words = []
    for word in words:
        new_words.append(word)
        if random.random() < noise_rate:
            new_words.append(random.choice(noise_words))
    return ' '.join(new_words)


def add_complexity(text):
    adjectives = ["beautiful", "strange", "mysterious", "colorful", "elegant"]
    adverbs = ["quickly", "silently", "gracefully", "slowly", "boldly"]

    sentences = text.split('. ')
    new_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 1:
            pos = random.randint(1, len(words) - 1)
            word = words[pos]
            if random.random() > 0.5:
                words.insert(pos, random.choice(adjectives))
            else:
                words.insert(pos, random.choice(adverbs))
        new_sentences.append(' '.join(words))
    return '. '.join(new_sentences)


def rewrite_text(sentence, num_replacements=10):
    """随机替换句子中的多个单词的同义词"""

    sentence = shuffle_sentences(sentence)
    sentence = add_noise(sentence)
    sentence = add_complexity(sentence)

    if num_replacements <= 0:
        return sentence

    words = sentence.split()
    if len(words) < num_replacements:
        num_replacements = len(words)

        # 随机选择不同单词的索引进行替换
    word_indices = sample(range(len(words)), num_replacements)

    for index in word_indices:
        word = words[index]
        synonyms = get_synonyms(word)
        if synonyms:
            # 随机选择一个同义词进行替换
            replacement = choice(synonyms)
            words[index] = replacement
    return ' '.join(words)


'''
sentence = "The quick brown fox jumps over the lazy dog."

# 替换句子中的一个单词  
replaced_sentence = replace_random_word_in_sentence(sentence)
print(replaced_sentence)
'''
# 注意：每次运行都可能得到不同的替换结果
'''
import nltk
from nltk.corpus import wordnet

# 下载所需的nltk数据（第一次运行时需要）
nltk.download('wordnet')
nltk.download('omw-1.4')


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def rewrite_text(text):
    words = text.split()
    rewritten_words = []

    for word in words:
        synonyms = get_synonyms(word)
        if synonyms and wordnet.synsets(word):
            rewritten_words.append(synonyms.pop())  # 替换为第一个同义词
        else:
            rewritten_words.append(word)

    return ' '.join(rewritten_words)


original_text = "AI technology is rapidly advancing, transforming many industries."
rewritten_text = rewrite_text(original_text)
print("Original Text:", original_text)
print("Rewritten Text:", rewritten_text)
'''
