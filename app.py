"""import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

#CHECKING IF TOKEN WAS LOADED
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Make sure it's in the .env file.")

#defining api url and headers
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}
print("Token used:", HF_TOKEN)


#function to call the Hugging Face API
def generate_image(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs":prompt})
    
    if response.status_code == 200:
        with open("generated_image.png", "wb") as f:
            f.write(response.content)
        print("Image generated and saved as 'generated_image.png'")
    else:
        print("Failed to generate image:", response.status_code, response.text)

# input prompt
generate_image("A ghost holding a coffee cup")
"""
import torch
from diffusers import StableDiffusionPipeline
import streamlit as st
from PIL import Image
import io

st.title("Stable Diffusion Image Generation")

prompt = st.text_input("Enter your prompt:", "A cute ghost holding a coffee cup")

if st.button("Generate Image"):
    with st.spinner("Generating image... Please wait beautiful."):
        # Load the pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32,
            use_safetensors=True
        )

        pipe = pipe.to("cpu")

        # Generate the image
        image = pipe(prompt).images[0]

        #show image
        st.image(image, caption="Generated Image", use_column_width=True)

        #Download button
        buf=io.BytesIO()
        image.save(buf, format="PNG")
        byte_im=buf.getvalue()
        st.download_button(label="Download Image",data=byte_im, file_name="generated_image.png", mime="image/png")



