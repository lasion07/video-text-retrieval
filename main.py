
# Import dependences
import cv2
import numpy as np
import os
import torch
import time
import streamlit as st

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

# Function to convert one video to images
@st.cache_data
def video2image(video_path, frame_rate=1.0, size=224):
    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)

    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        images = np.zeros([3, size, size], dtype=np.float32)
        print("ERROR: problem reading video file: ", video_path)
    else:
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        interval = fps / frame_rate
        frames_idx = np.floor(np.arange(start_sec*fps, end_sec*fps, interval))
        ret = True
        images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)

        for i, idx in enumerate(frames_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES , idx)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = i
            images[i,:,:,:] = preprocess(size, Image.fromarray(frame).convert("RGB"))

        images = images[:last_frame+1]
    cap.release()
    video_frames = torch.tensor(images)
    return video_frames

@st.cache_data
def load_vision_model(model_name="Searchium-ai/clip4clip-webvid150k"):
    vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    vision_model.eval()
    return vision_model

@st.cache_data
def load_text_model(model_name="Searchium-ai/clip4clip-webvid150k"):
    text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model.eval()
    return text_model, tokenizer

@st.cache_data
def embedding_video(video_path):
    vision_model = load_vision_model()

    video_frames = video2image(video_path)
    video_frames = video_frames.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    visual_output = vision_model(video_frames)

    # Normalizing the embeddings and calculating mean between all embeddings.
    visual_output = visual_output["image_embeds"]
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    visual_output = torch.mean(visual_output, dim=0)
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    return visual_output

@st.cache_data
def embedding_storage(video_storage_path):
    video_metadata_list = []

    with torch.no_grad():
        for i, file in enumerate(os.listdir(video_storage_path)):
            if file.endswith(".mp4"):
                video_path = os.path.join(video_storage_path, file)
                video_embedding = embedding_video(video_path)
                video_metadata_list.append({"video_path": video_path, "video_embedding": video_embedding})
    
    return video_metadata_list

@st.cache_data
def embedding_text(raw_text_input):
    text_model, tokenizer = load_text_model()
    text_input = tokenizer(raw_text_input, return_tensors="pt").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    text_output = text_model(text_input["input_ids"])
    text_output = text_output["text_embeds"]
    text_output = text_output / text_output.norm(dim=-1, keepdim=True)

    return text_output

if __name__ == '__main__':
    st.title('Video-Text Retrieval')

    # Embedding video storage
    st.header('Scanning video storage')
    video_storage_path = "data"
    video_metadata_list = embedding_storage(video_storage_path)
    st.success(f'Embedding {len(video_metadata_list)} videos successfully!')

    # Find video by text
    st.header('Find video by text')

    ## Set distance
    distance = st.slider('Min distance', 0.0, 1.0, 0.15)

    ## Input text
    raw_text_input = st.text_input('Text input', 'What is the capital of France?')
    if st.button('Search', use_container_width=True):
        text_embedding = embedding_text(raw_text_input)

        # Find video by text
        similarities = []
        for video_metadata in video_metadata_list:
            video_embedding = video_metadata["video_embedding"]
            similarity = torch.cosine_similarity(text_embedding, video_embedding)
            if similarity.item() < distance:
                continue

            similarities.append([similarity.item(), video_metadata["video_path"]])

        if len(similarities) == 0:
            st.error('No video found')
            st.stop()

        # The most relevant videos
        st.header(f'Results for "{raw_text_input}"')
        st.success(f'Found {len(similarities)} videos')
        for i, (similarity, video_path) in enumerate(sorted(similarities, key=lambda x: x[0], reverse=True)):
            # st.info(f'[{i+1}] {os.path.basename(video_path)}')
            st.subheader(f'{os.path.basename(video_path)}')
            st.text(f'Video path: {video_path} - Similarity: {round(similarity, 2)}')
            st.video(video_path)

