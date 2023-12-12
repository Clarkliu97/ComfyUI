import json
from urllib import request
import random

# Global Variables
STEPS = 60

# Initialize Models
# Load Checkpoint 
CHECKPOINT_NAME = "lineLibraryHand_v10.safetensors"
# Load LoRA 
LORA_NAME = "2023-12-08-2.safetensors" # strength_model = 1.0, strength_clip = 1.0
# CLIP Set Last Layer = -2
# Positive Text Prompt
POSITIVE_PROMPT = "a masterpiece"
# Negative Text Prompt
NEGATIVE_PROMPT = "bad hands"

# Create Background
# CLIP Text Encode (Prompt)
BACKGROUND_PROMPT = "a background"
# Empty Latent Image
BACKGROUND_WIDTH = 1280
BACKGROUND_HEIGHT = 720
# KSampler (Advanced)
BACKGROUND_END_AT_STEP = 10
BACKGROUND_ADD_NOISE = "enable"
BACKGROUND_CFG = 5.0
BACKGROUND_SAMPLER_NAME = "dpmpp_2m"
BACKGROUND_SCHEDULER = "karras"
BACKGROUND_START_AT_STEP = 0
BACKGROUND_RETURN_NOISE = "enable"

# Create Entity (Character)
# CLIP Text Encode (Prompt)
ENTITY_PROMPT = "a character"
# Empty Latent Image
ENTITY_WIDTH = 256
ENTITY_HEIGHT = 512
# KSampler (Advanced)
ENTITY_END_AT_STEP = 10
ENTITY_ADD_NOISE = "enable"
ENTITY_CFG = 5.0
ENTITY_SAMPLER_NAME = "dpmpp_2m"
ENTITY_SCHEDULER = "karras"
ENTITY_START_AT_STEP = 0
ENTITY_RETURN_NOISE = "enable"

# Combine Latent Images
# MultiLatentComposite

# Render Image
# KSampler (Advanced)
RENDER_END_AT_STEP = 10000
RENDER_ADD_NOISE = "disable"
RENDER_CFG = 5.0
RENDER_SAMPLER_NAME = "dpmpp_2m"
RENDER_SCHEDULER = "karras"
RENDER_START_AT_STEP = 10
RENDER_RETURN_NOISE = "disable"
# VAE Decoder
# Save Image
FILE_NAME_PREFIX = "test-"

# global parameters
id = 0
nodes = {}
extra_data = {}

endpoints = {}
entity_num = 0
entity_endpoints = {}

multi_latent_composite_id = -1
need_seed_ids = []

def load_checkpoint():
    global id
    global endpoints
    # Load Checkpoint 
    checkpoint_loder = {str(id): {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT_NAME}}}
    endpoints["model"] = [str(id), 0]
    endpoints["clip"] = [str(id), 1]
    endpoints["VAE"] = [str(id), 2]
    id += 1
    nodes.update(checkpoint_loder)

def load_lora(): 
    global id
    global endpoints
    # TODO: Load LoRA
    pass
    
def clip_set_last_layer(stop_at_clip_layer = -2):
    global id
    global endpoints
    csll = {str(id): {"class_type": "CLIPSetLastLayer", "inputs": {"clip": endpoints["clip"], "stop_at_clip_layer": stop_at_clip_layer}}}
    endpoints["clip"] = [str(id), 0]
    id += 1
    nodes.update(csll)

def positive_text_prompt(positive_prompt = POSITIVE_PROMPT):
    global id
    global endpoints
    ptp = {str(id): {"class_type": "CLIPTextEncode", "inputs": {"clip": endpoints["clip"], "text": positive_prompt}}}
    endpoints["positive_condition"] = [str(id), 0]
    id += 1
    nodes.update(ptp)

def negative_text_prompt(negative_prompt = NEGATIVE_PROMPT):
    global id
    global endpoints
    ntp = {str(id): {"class_type": "CLIPTextEncode", "inputs": {"clip": endpoints["clip"], "text": negative_prompt}}}
    endpoints["negative_condition"] = [str(id), 0]
    id += 1
    nodes.update(ntp)

def create_background(background_prompt = BACKGROUND_PROMPT, background_width = BACKGROUND_WIDTH, background_height = BACKGROUND_HEIGHT, background_end_at_step = BACKGROUND_END_AT_STEP, background_add_noise = BACKGROUND_ADD_NOISE, background_cfg = BACKGROUND_CFG, background_sampler_name = BACKGROUND_SAMPLER_NAME, background_scheduler = BACKGROUND_SCHEDULER, background_start_at_step = BACKGROUND_START_AT_STEP, background_return_noise = BACKGROUND_RETURN_NOISE):
    global id
    global endpoints
    # Create Background
    # CLIP Text Encode (Prompt)
    ctep = {str(id): {"class_type": "CLIPTextEncode", "inputs": {"clip": endpoints["clip"], "text": background_prompt}}}
    endpoints["background_condition"] = [str(id), 0]
    id += 1
    nodes.update(ctep)
    # Empty Latent Image
    eli = {str(id): {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": background_height, "width": background_width}}}
    endpoints["background_latent_image"] = [str(id), 0]
    id += 1
    nodes.update(eli)

    # KSampler (Advanced)
    ksa = {str(id): {"class_type": "KSamplerAdvanced", "inputs": {
        "add_noise": background_add_noise,
        "noise_seed": 0,
        "steps": STEPS,
        "cfg": background_cfg,
        "sampler_name": background_sampler_name,
        "scheduler": background_scheduler,
        "start_at_step": background_start_at_step,
        "end_at_step": background_end_at_step,
        "return_with_leftover_noise": background_return_noise,
        "model": endpoints["model"],
        "positive": endpoints["background_condition"],
        "negative": endpoints["negative_condition"],
        "latent_image": endpoints["background_latent_image"]
    }}}
    endpoints["background_latent_image"] = [str(id), 0]
    need_seed_ids.append(str(id))
    id += 1
    nodes.update(ksa)

def create_entity(entity_prompt = ENTITY_PROMPT, entity_width = ENTITY_WIDTH, entity_height = ENTITY_HEIGHT, entity_end_at_step = ENTITY_END_AT_STEP, entity_add_noise = ENTITY_ADD_NOISE, entity_cfg = ENTITY_CFG, entity_sampler_name = ENTITY_SAMPLER_NAME, entity_scheduler = ENTITY_SCHEDULER, entity_start_at_step = ENTITY_START_AT_STEP, entity_return_noise = ENTITY_RETURN_NOISE):
    global id
    global endpoints
    global entity_num
    # Create Entity (Character)
    # CLIP Text Encode (Prompt)
    ctep = {str(id): {"class_type": "CLIPTextEncode", "inputs": {"clip": endpoints["clip"], "text": entity_prompt}}}
    entity_endpoints[str(entity_num) + "_condition"] = [str(id), 0]
    id += 1
    nodes.update(ctep)
    # Empty Latent Image
    eli = {str(id): {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": entity_height, "width": entity_width}}}
    entity_endpoints[str(entity_num) + "_latent_image"] = [str(id), 0]
    id += 1
    nodes.update(eli)
    # KSampler (Advanced)
    ksa = {str(id): {"class_type": "KSamplerAdvanced", "inputs": {
        "add_noise": entity_add_noise,
        "noise_seed": 0,
        "steps": STEPS,
        "cfg": entity_cfg,
        "sampler_name": entity_sampler_name,
        "scheduler": entity_scheduler,
        "start_at_step": entity_start_at_step,
        "end_at_step": entity_end_at_step,
        "return_with_leftover_noise": entity_return_noise,
        "model": endpoints["model"],
        "positive": entity_endpoints[str(entity_num) + "_condition"],
        "negative": endpoints["negative_condition"],
        "latent_image": entity_endpoints[str(entity_num) + "_latent_image"]
    }}}
    entity_endpoints[str(entity_num) + "_latent_image"] = [str(id), 0]
    need_seed_ids.append(str(id))
    id += 1
    nodes.update(ksa)
    entity_num += 1

def multi_latent_composite():
    global id
    global endpoints
    global entity_num
    global multi_latent_composite_id
    # Combine Latent Images
    # MultiLatentComposite
    inputs = {
        "MultiLatentComposite-Canvas": None,
        "index": 0, 
        "x": 0,
        "y": 0,
        "feather": 0,
        "samples_to": endpoints["background_latent_image"],
    }
    for i in range(entity_num):
        inputs["samples_from" + str(i)] = entity_endpoints[str(i) + "_latent_image"]
    mlc = {str(id): {"class_type": "MultiLatentComposite", "inputs": inputs}}
    endpoints["canvas"] = [str(id), 0]
    multi_latent_composite_id = id
    id += 1
    nodes.update(mlc)

def render_image(render_end_at_step = RENDER_END_AT_STEP, render_add_noise = RENDER_ADD_NOISE, render_cfg = RENDER_CFG, render_sampler_name = RENDER_SAMPLER_NAME, render_scheduler = RENDER_SCHEDULER, render_start_at_step = RENDER_START_AT_STEP, render_return_noise = RENDER_RETURN_NOISE):
    global id
    global endpoints
    # Render Image
    # KSampler (Advanced)
    ksa = {str(id): {"class_type": "KSamplerAdvanced", "inputs": {
        "add_noise": render_add_noise,
        "noise_seed": 0,
        "steps": STEPS,
        "cfg": render_cfg,
        "sampler_name": render_sampler_name,
        "scheduler": render_scheduler,
        "start_at_step": render_start_at_step,
        "end_at_step": render_end_at_step,
        "return_with_leftover_noise": render_return_noise,
        "model": endpoints["model"],
        "positive": endpoints["positive_condition"],
        "negative": endpoints["negative_condition"],
        "latent_image": endpoints["canvas"]
    }}}
    endpoints["canvas"] = [str(id), 0]
    need_seed_ids.append(str(id))
    id += 1
    nodes.update(ksa)
    # VAE Decoder
    vd = {str(id): {"class_type": "VAEDecode", "inputs": {"samples": endpoints["canvas"], "vae": endpoints["VAE"]}}}
    endpoints["image"] = [str(id), 0]
    id += 1
    nodes.update(vd)
    # Save Image
    si = {str(id): {"class_type": "SaveImage", "inputs": {"images": endpoints["image"], "filename_prefix": FILE_NAME_PREFIX}}}
    id += 1
    nodes.update(si)

def queue_prompt(prompt, extra_data):
    p = {"prompt": prompt, 
         "extra_data": extra_data}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)

def main(): 
    load_checkpoint()
    clip_set_last_layer()
    positive_text_prompt()
    negative_text_prompt()
    create_background()
    create_entity()
    create_entity()
    # create_entity()
    multi_latent_composite()
    render_image()
    print(nodes)
    print(endpoints)
    print(entity_endpoints)

    # Set Random Seed
    rnd_int = random.randint(0, 100000)
    for id in need_seed_ids:
        nodes[id]["inputs"]["noise_seed"] = rnd_int

    # TODO: Generate Prompts

    extra_data = {"extra_pnginfo": {"workflow": {"nodes": [{"id": multi_latent_composite_id, "properties": {"values": [[840,208,80],[336,200,80]]}}]}}}
    queue_prompt(nodes, extra_data)

if __name__ == "__main__":
    main()