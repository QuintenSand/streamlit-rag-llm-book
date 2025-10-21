# app
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

st.header("Hands-On Large Language Models book RAG")

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What do you want to know about the book?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
    
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response = st.write(output)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

#prompt = st.chat_input("Give me a short introduction to large language model.")
#if prompt:
#    messages = [
#    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#    {"role": "user", "content": prompt}
#    ]
#    text = tokenizer.apply_chat_template(
#        messages,
#        tokenize=False,
#        add_generation_prompt=True
#    )
#    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#    
#    generated_ids = model.generate(
#        **model_inputs,
#        max_new_tokens=512
#    )
#    
#    generated_ids = [
#        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#    ]
 #   
 #   response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

 #   st.write(response)