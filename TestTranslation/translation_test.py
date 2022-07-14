from TestTranslation.translation import *

transformer.evaluate(train_ds)

test_eng_texts = [pair[0] for pair in test_pairs]
input_sentence = "This is a test."
translated = decode_sequence(input_sentence)
print(input_sentence)
print(translated)

for _ in range(30):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)
    print(input_sentence)
    print(translated)
