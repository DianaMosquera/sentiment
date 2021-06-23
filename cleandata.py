# from polyglot.text import Text
from polyglot.detect import Detector
import icu
import string
import pandas as pd
# from polyglot.downloader import downloader
# import numpy as np
# import nltk
import re
import csv


def strip_links(text):
    link_regex = re.compile(
        "((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)",
        re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text


def strip_all_entities(text):
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator, ' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word and word[0] not in entity_prefixes:
            words.append(word)
    return ' '.join(words)


def detect_language(x):
    try:
        poly_obj = Detector(x, quiet=True)
        text_lang = icu.Locale.getDisplayName(poly_obj.language.locale)
    except Exception as e:
        print(e)
        text_lang = 'unknown'

    return text_lang


def read_age_lexica(directory):
    """reads in the raw text

        Args:
            directory: location of the age lexicon file
        Returns:
            df: dataframe containing words and weights from age lexicon
    """

    age_lexica = {}
    with open(directory, mode='r') as infile:
        reader = csv.DictReader(infile)
        for data in reader:
            weight = float(data['weight'])
            term = data['term']
            age_lexica[term] = weight

    del age_lexica['_intercept']
    return age_lexica


def read_gender_lexica(directory):
    """reads in the raw text

        Args:
            directory: location of the age lexicon file
        Returns:
            df: dataframe containing words and weights from gender lexicon
    """

    gender_lexica = {}
    with open(directory, mode='r') as infile:
        reader = csv.DictReader(infile)
        for data in reader:
            weight = float(data['weight'])
            term = data['term']
            gender_lexica[term] = weight

    del gender_lexica['_intercept']
    return gender_lexica


def age_predictor(text, age_lexica, age_intercept):
    """cleans the raw text
        Args:
            text: social media post based on which age needs to be inferred
            age_lexica: words and weights pre-calculated
            age_intercept: mean age
        Returns:
            age: predicted age

    """
    ###Test if text contains nulls###
    #    if type(text) != str: assert np.isnan(text) == False, 'Text contains nulls'

    words = text.split()
    text_scores = {}
    for word in words:
        text_scores[word] = text_scores.get(word, 0) + 1
    age = 0
    words_count = 0
    for word, count in text_scores.items():
        if word in age_lexica:
            words_count = words_count + count
            age = age + (count * age_lexica[word])

    try:
        age = (age / words_count) + age_intercept
    except Exception as e:
        print(e)
        age = "unknown"

    assert age_intercept == 23.2188604687, 'Age Intercept should be equal to 23.2188604687'

    return age


def gender_predictor(text, gender_lexica, gender_intercept):
    words = text.split()

    text_scores = {}
    for word in words:
        text_scores[word] = text_scores.get(word, 0) + 1

    gender = 0
    words_count = 0
    for word, count in text_scores.items():
        if word in gender_lexica:
            words_count += count
            gender += count * gender_lexica[word]

    try:
        gender = gender / words_count + gender_intercept
    except Exception as e:
        print(e)
        gender = "unknown"

    assert gender_intercept == -0.06724152, "Gender Intercept should be equal to -0.06724152"

    return gender


def map_age_value(age_value):
    if age_value != "unknown":
        if age_value <= 21:
            return "less than 21"
        elif 21 < age_value <= 26:
            return "21-25"
        elif 26 < age_value <= 36:
            return "26-35"
        elif 36 < age_value <= 46:
            return "36-45"
        elif 46 < age_value <= 56:
            return "46-55"
        else:
            return "greater than 55"

    return "unknown"


def map_gender_value(gender_value):
    if gender_value != "unknown":
        if gender_value >= 0:
            return "female"
        else:
            return "male"

    return "unknown"


if __name__ == '__main__':
    PROJECT_DIR = "/home/jpvazque/Documents/TRABAJO/sentiment-analysis/age_gender_predictor/"

    AGE_LEXICA = read_age_lexica(PROJECT_DIR + "emnlp14age.csv")
    GENDER_LEXICA = read_gender_lexica(PROJECT_DIR + "emnlp14gender.csv")

    dataset_unified = pd.read_excel("datasetPostUnified.xlsx", engine="openpyxl")

    dataset_unified["Clean_Conversation_Stream"] = dataset_unified["Conversation_Stream_x"].apply(
        lambda raw_text: strip_all_entities(strip_links(raw_text))
    )

    dataset_unified["language"] = \
        dataset_unified["Clean_Conversation_Stream"].apply(
            lambda clean_text: detect_language(clean_text)
        )

    dataset_unified["user_age"] = \
        dataset_unified["Clean_Conversation_Stream"].apply(
            lambda clean_text: map_age_value(
                age_predictor(
                    clean_text.lower(),
                    AGE_LEXICA,
                    age_intercept=23.2188604687
                )
            )
        )

    dataset_unified.loc[dataset_unified["language"] != "English", "user_age"] = "unknown"

    dataset_unified["user_gender"] = \
        dataset_unified["Clean_Conversation_Stream"].apply(
            lambda clean_text: map_gender_value(
                gender_predictor(
                    clean_text.lower(),
                    GENDER_LEXICA,
                    gender_intercept=-0.06724152
                )
            )
        )

    dataset_unified.loc[dataset_unified["language"] != "English", "user_gender"] = "unknown"

    dataset_unified.to_excel("datasetPostAgeGenderPrediction.xlsx", index=False)
