#!/usr/bin/env python3
#
# Example for vision/language model inference on continuous video streams
# python3 -m nano_llm.vision.video_openbmb  
#   --video-input /dev/video0  --video-input-width 640 
#   --video-input-height 480    --video-output display://0 
#
#
#
import time
import logging
from PIL import Image
from nano_llm import NanoLLM, ChatHistory, remove_special_tokens
from nano_llm.utils import ArgParser, load_prompts, wrap_text
from nano_llm.plugins import VideoSource, VideoOutput

from termcolor import cprint
from jetson_utils import cudaMemcpy, cudaToNumpy, cudaFont
from transformers import AutoModel, AutoTokenizer #, BitsAndBytesConfig, AutoConfig





model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True,low_cpu_mem_usage = True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)


model.eval()


# parse args and set some defaults
parser = ArgParser(extras=ArgParser.Defaults + ['video_input', 'video_output'])
#parser.add_argument("--max-images", type=int, default=8, help="the number of video frames to keep in the history")
args = parser.parse_args()


question = 'Describe image concisely'
msgs = [{'role': 'user', 'content': question}]



# open the video stream
num_images = 0
last_image = None
last_text = ''

def on_video(image):
    global last_image
    last_image = cudaMemcpy(image)
    if last_text:
        #font_text = remove_special_tokens(last_text)
        wrap_text(font, image, text=last_text, x=5, y=5, color=(120,215,21), background=font.Gray50)
    video_output(image)
    
video_source = VideoSource(**vars(args), cuda_stream=0)
video_source.add(on_video, threaded=False)
video_source.start()

video_output = VideoOutput(**vars(args))
video_output.start()

font = cudaFont()

# apply the prompts to each frame
while True:
    if last_image is None:
        continue
    
    image_np = cudaToNumpy(last_image)
    image = Image.fromarray(image_np).convert('RGB')
    last_image = None
    num_images += 1
    image.show()

    res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
    )
    last_text = res
    print(res)
        
    if video_source.eos:
        video_output.stream.Close()
        break
