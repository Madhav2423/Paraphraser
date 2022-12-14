from flask import Flask, jsonify, request, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pickle
import streamlit as st
app= Flask(__name__)


model_main= pickle.load(open('t5-small.pkl', 'rb'))

tokenizer = T5Tokenizer.from_pretrained("hetpandya/t5-small-tapaco")
model = T5ForConditionalGeneration.from_pretrained("hetpandya/t5-small-tapaco")

def get_paraphrases(sentence, prefix="paraphrase: ", n_predictions=5, top_k=120, max_length=256,device="cpu"):
        text = prefix + sentence + " </s>"
        encoding = tokenizer.encode_plus(
            text, pad_to_max_length=True, return_tensors="pt"
        )
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding[
            "attention_mask"
        ].to(device)

        model_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            max_length=max_length,
            top_k=top_k,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=n_predictions,
        )

        outputs = []
        for output in model_output:
            generated_sent = tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            if (
                generated_sent.lower() != sentence.lower()
                and generated_sent not in outputs
            ):
                outputs.append(generated_sent)
        return outputs
def first(x):
    paraphrases = get_paraphrases(x, n_predictions=1)
    ans=''
    for sent in paraphrases:
        ans+=sent
    return ans

#     app.run(debug= True)
# x="bennett university is good"
# print(first(x))

st.title("Paraphraser")
html_temp = """
<div style="background-color:red;padding:5px">
<h2 style="color:white;text-align:center;"> Let's Go </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
result=""
x = st.text_input(
        "Enter the string you want to paraphrase : 👇"
        
    )
if st.button("Paraphrase"):
    
    a=first(x)
    
    st.write(a)
    # app.run(debug= True)

    

    