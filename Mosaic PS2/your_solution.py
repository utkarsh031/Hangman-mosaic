import string
import random
from collections import defaultdict, Counter
import numpy as np


file_path = "training.txt"
with open(file_path, "r") as file:
    dataset = file.read().splitlines()

training_set = []
test_set = []
unique_words = list(set(dataset))
unique_words = np.array(unique_words)
np.random.shuffle(unique_words)
unique_words = unique_words.tolist()

test_set = unique_words[:0]
training_set = unique_words[0:]
def unigram(corpus):
    unigram_counts = Counter()

    for word in corpus:
        for char in word:
            unigram_counts[char] += 1

    return unigram_counts
unigram_counts = unigram(training_set)
# Calculate bigram probability
def bigram_prob(key, char, bigram_counts):
    prev_word_counts = bigram_counts[key]
    total_counts = float(sum(prev_word_counts.values()))

    return prev_word_counts[char] / float(sum(prev_word_counts.values()))


# Calculate trigram probability
def trigram_prob(wi_2, wi_1, char, trigram_counts, bigram_counts):
    return (trigram_counts[wi_2 + wi_1][char] / float(bigram_counts[wi_2][wi_1]))



# Add $$$ to the front of a word
def fourgram_convert_word(word):
    return "$$$" + word

# Collect fourgram counts
def fourgram(corpus):
    fourgram_counts = Counter()
    trigram_counts = defaultdict(Counter)
    bigram_counts = defaultdict(Counter)

    for word in corpus:
        word = fourgram_convert_word(word)

        # Generate a list of fourgrams
        fourgram_list = zip(word[:-3], word[1:-2], word[2:-1], word[3:])

        # Generate a list of trigrams
        trigram_list = zip(word[:-2], word[1:-1], word[2:])

        # Generate a list of bigrams
        bigram_list = zip(word[:-1], word[1:])

        # Iterate over fourgrams
        for fourgram in fourgram_list:
            first, second, third, fourth = fourgram
            element = first + second + third + fourth
            fourgram_counts[element] += 1

        # Iterate over trigrams
        for trigram in trigram_list:
            first, second, third = trigram
            trigram_counts[first + second][third] += 1

        # Iterate over bigrams
        for bigram in bigram_list:
            first, second = bigram
            bigram_counts[first][second] += 1

    return fourgram_counts, trigram_counts, bigram_counts

# Update trigram and bigram counts for 4-gram model
fourgram_counts, trigram_counts_for_fourgram, bigram_counts_for_trigram = fourgram(training_set)


# Calculate 4-gram probability
def fourgram_prob(wi_3, wi_2, wi_1, char, fourgram_counts, trigram_counts, bigram_counts):
    # Calculate the trigram probability P(wi | wi-2, wi-1)
    trigram_probability = trigram_counts[wi_3 + wi_2][wi_1] / float(bigram_counts[wi_3][wi_2])

    # If the trigram count is zero, return zero probability
    if trigram_probability == 0:
        return 0

    # Calculate the 4-gram probability P(wi | wi-3, wi-2, wi-1)
    return fourgram_counts[wi_3 + wi_2 + wi_1 + char] / float(trigram_counts[wi_3 + wi_2][wi_1])

def four_gram_guesser(mask, guessed, bigram_counts=bigram_counts_for_trigram,
                      trigram_counts=trigram_counts_for_fourgram, fourgram_counts=fourgram_counts,
                          unigram_counts=unigram_counts):
    available = list(set(string.ascii_lowercase) - guessed)
    fourgram_probs = []
    mask = ['$', '$', '$'] + mask
    fourgram_lambda = 0.4
    trigram_lambda = 0.3
    bigram_lambda = 0.2
    unigram_lambda = 0.1

    for char in available:
        char_prob = 0
        for index in range(len(mask)):
            if index == 0 and mask[index] == '_':
                char_prob += ((fourgram_lambda * fourgram_prob('$', '$', '$', char, fourgram_counts, trigram_counts,bigram_counts))
                              + (trigram_lambda * trigram_prob('$','$',char, trigram_counts, bigram_counts))
                              + (bigram_lambda * bigram_prob('$', char, bigram_counts))
                              + unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values())))

            if index == 1 and mask[index] == '_':
                # If the previous word has been guessed, apply trigram
                if not mask[index - 1] == '_':
                    char_prob += ((fourgram_lambda * fourgram_prob('$', '$', mask[index - 1], char, fourgram_counts, trigram_counts,bigram_counts))
                                  + (trigram_lambda * trigram_prob('$',mask[index - 1],char, trigram_counts, bigram_counts))
                                  + (bigram_lambda * bigram_prob(mask[index - 1], char, bigram_counts))
                                  + unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values())))
                # If the previous word has not been guessed, apply unigram
                else:
                    char_prob += unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values()))

            if index == 2 and mask[index] == '_':
                # If the previous 2 word has been guessed, apply fourgram
                if not mask[index - 1] == '_' and not mask[index-2] == '_':
                    char_prob += ((fourgram_lambda * fourgram_prob('$', mask[index - 2], mask[index - 1], char, fourgram_counts,
                                                                   trigram_counts, bigram_counts))
                                  + (trigram_lambda * trigram_prob(mask[index - 2],mask[index - 1],char, trigram_counts, bigram_counts))
                                  + (bigram_lambda * bigram_prob(mask[index - 1], char, bigram_counts))
                                  + unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values())))
                # if index-1 is guessed and index-2 is not , apply bigram
                elif not mask[index-1] == '_':
                    char_prob += bigram_lambda * bigram_prob(mask[index - 1], char, bigram_counts)
                # If the previous word has not been guessed, apply unigram
                else:
                    char_prob += unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values()))

            elif mask[index] == '_':
                # if wi_3 , wi_2 , wi_1 all are guessed , apply fourgram
                if not mask[index - 1] == '_' and not mask[index-2] == '_' and not mask[index-3] == '_':
                    char_prob += ((fourgram_lambda * fourgram_prob(mask[index-3], mask[index - 2], mask[index-1], char, fourgram_counts, trigram_counts,bigram_counts))
                                  + (trigram_lambda * trigram_prob(mask[index - 2],mask[index - 1],char, trigram_counts, bigram_counts))
                                  + (bigram_lambda * bigram_prob(mask[index - 1], char, bigram_counts))
                                  + unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values())))
                # if wi_3 is not guessed but wi_2 and wi_1 are guessed , use trigram
                elif not mask[index - 1] == '_' and not mask[index-2] == '_':
                    # char_prob += trigram_lambda * trigram_prob(mask[index - 2], mask[index-1], char,trigram_counts, bigram_counts)
                    char_prob += ((trigram_lambda * trigram_prob(mask[index - 2],mask[index - 1],char, trigram_counts, bigram_counts))
                                  + (bigram_lambda * bigram_prob(mask[index - 1], char, bigram_counts))
                                  + unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values())))
                # if wi_3, wi_2 is not guessed but  wi_1 are guessed , use bigram
                elif not mask[index - 1] == '_':
                    char_prob += bigram_lambda * bigram_prob(mask[index - 1], char, bigram_counts)
                else:
                    char_prob += unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values()))
            else:
                continue
        fourgram_probs.append(char_prob)

    # Return the max probability of char
    return available[fourgram_probs.index(max(fourgram_probs))]



def suggest_next_letter_sol(displayed_word, guessed_letters):
    """summary

    This function takes in the current state of the game and returns the next letter to be guessed.
    displayed_word: str: The word being guessed, with underscores for unguessed letters.
    guessed_letters: list: A list of the letters that have been guessed so far.
    Use python hangman.py to check your implementation.
    """
    guessed_letters_set = set(guessed_letters)
    return four_gram_guesser(list(displayed_word), guessed_letters_set)
    ################################################
    ################################################
    ################################################
                #   Your Code HERE. #
                #                   #
                #                   #
                #                   #
                #                   #
                #                   #
                #                   #
    ################################################
    ################################################
    ################################################
    raise NotImplementedError
