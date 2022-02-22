from predict import unzip_model, load_model, predict_labels

# get model_zip_path
# model_zip_path = "/Users/zhenshanjin/Documents/Belmont/sandy/UtilityDaisies/BartLargeMultiNLI/tmp/bart-large-mnli.zip"
model_path = "/pebble_source/88fc1fd1-8729-43b9-aa71-34108fd3f3f8/model"
# model_path = unzip_model(model_zip_path)
classifier = load_model(model_path)

def compute(text, candidate_labels, is_multi_labels="false"):
    res = predict_labels(text, candidate_labels, classifier, is_multi_labels)
    
    return {"result": res}

if __name__ == "__main__":
    text = "one day I will see the world"
    candidate_labels = "travel, cooking, dancing"
    print(compute(text, candidate_labels))
    print(compute("one day I will see the world", "travel, cooking, dancing, exploration", "true")) 