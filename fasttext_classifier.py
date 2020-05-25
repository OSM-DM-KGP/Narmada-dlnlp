from warnings import warn
import fasttext
import pudb

warn("Please run `python3 prepare_tweets_fasttext.py nepal` before running this script")

model = fasttext.train_supervised(input = "DATA_2/INPUT/ft/train_nepal_processed")
model.save_model("saved_models/model_fasttext.bin")

k = model.test("DATA_2/INPUT/ft/test_nepal_processed")
print(k)
pu.db