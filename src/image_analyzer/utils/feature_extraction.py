import cv2
import io
import math
import numpy as np
import os
import pandas as pd
import platform
import pytesseract
import pywt
import tensorflow as tf
import torch
import traceback

from ..utils.helper_functions import download_weights
from langdetect import detect, LangDetectException
from mtcnn import MTCNN
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from skimage import io, filters, measure
from skimage.color import label2rgb
from skimage.io import imread
from skimage.measure import regionprops, label
from skimage.segmentation import felzenszwalb, slic
from torchvision import models, transforms
from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoProcessor, BlipProcessor, BlipForConditionalGeneration, GenerationConfig
from ultralytics import YOLO

def new_feature_template(self, df_images):
    """
    <Give a short description of what this function computes and how it is structured on a high-level>

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added feature columns:
        - new_feature: Description of your new feature
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filePath'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image using cv2
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue

            # Do your magic here - THANK YOU SO MUCH!!!
            new_feature = True

            # Save your feature to the dataframe
            df.loc[idx, 'new_feature'] = new_feature

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_new_feature'] = error
            continue

    return df

def calculate_aesthetic_scores(self, df_images):
    """
    Calculates the aesthetic scores for a list of images using a MobileNet-based NIMA model.
    
    For each image, the following steps are performed:
      - Load and resize the image to 224x224.
      - Convert the image from BGR to RGB and normalize pixel values to the range [0, 1].
      - The pre-trained MobileNet-based NIMA model predicts a probability distribution over 10 score bins.
      - The aesthetic score is computed as the weighted sum: sum_{i=1}^{10} (i * p_i),
        where p_i is the predicted probability for score bin i.
    
    :param self: IA object
    :param df_images: DataFrame with a column 'filePath' containing paths to image files.
    :return: DataFrame with an additional column 'nima_score' containing the aesthetic scores.
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    try:
        # Get parameters
        weight_filename = self.config.get("features", {}).get("calculate_aesthetic_scores", {}).get("parameters", {}).get("weight_filename")
        weight_url = self.config.get("features", {}).get("calculate_aesthetic_scores", {}).get("parameters", {}).get("weight_url")

        # Download weights if needed
        try:
            download_weights(
                weight_filename=weight_filename,
                weight_url=weight_url
            )
        except Exception as e:
            print(f"Error downloading weights: {str(e)}")
            return df

        # Load model
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'weights', weight_filename)
            
            base_model = tf.keras.applications.MobileNet(
                input_shape=(224, 224, 3),
                include_top=False,
                pooling='avg',
                weights=None
            )
            x = base_model.output
            x = tf.keras.layers.Dropout(0.75)(x)
            x = tf.keras.layers.Dense(10, activation='softmax')(x)
            model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
            model.load_weights(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return df

        # Process images
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Read and process image
                image = cv2.imread(image_path)
                if image is None:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Preprocess image
                image_resized = cv2.resize(image, (224, 224))
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image_norm = image_rgb.astype("float32") / 255.0
                image_input = np.expand_dims(image_norm, axis=0)

                # Predict aesthetic score
                predictions = model.predict(image_input, verbose=False)
                p = predictions[0]
                aesthetic_score = sum((i + 1) * p[i] for i in range(len(p)))
                df.loc[idx, 'nima_score'] = aesthetic_score

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_nima'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in NIMA setup: {str(e)}")
        return df

def calculate_hue_proportions(self, df_images):
    """
    Calculate the proportions of warm and cold hues in images using HSV color space.
    Warm hues are defined as those outside 30-110° in HSV (reds, oranges, yellows),
    while cold hues are those within 30-110° (greens, blues).
    
    Following Zhang & Luo (Wang et al., 2013): cool = hue ∈ [30°, 110°], warm = complement.
    OpenCV HSV uses H ∈ [0,179] (≈ degrees/2), so cool = H ∈ [15, 55].

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added columns:
            - hues_warm: proportion of warm-hue pixels (0-1)
            - hues_cold: proportion of cold-hue pixels (0-1)
    :note: Proportions sum to 1 (excluding any neutral/grayscale pixels)
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filePath'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue
            
            # Convert the image to HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Extract the hue channel
            hue_channel = hsv_image[:, :, 0]
            
            # Define masks for cold and warm hues using OpenCV HSV scale
            # OpenCV H ∈ [0,179] maps to degrees/2, so 30°-110° becomes 15-55
            cold_hue_mask = (hue_channel >= 15) & (hue_channel <= 55)
            warm_hue_mask = ~cold_hue_mask  # Invert cold hue mask to get warm hue mask
            
            # Count the number of pixels in each range
            cold_pixel_count = np.sum(cold_hue_mask)
            warm_pixel_count = np.sum(warm_hue_mask)
            total_pixels = cold_pixel_count + warm_pixel_count
            
            # Calculate proportions (avoid division by zero)
            if total_pixels == 0:
                warm_proportion = 0.0
                cold_proportion = 0.0
            else:
                warm_proportion = warm_pixel_count / total_pixels
                cold_proportion = cold_pixel_count / total_pixels
            
            df.loc[idx, 'hues_warm'] = warm_proportion
            df.loc[idx, 'hues_cold'] = cold_proportion

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_hue'] = error
            continue

    return df

def calculate_image_clarity(self, df_images):
    """
    Calculate the clarity score for each image by measuring the proportion of high-brightness pixels.
    Following Zhang & Luo's method: normalize the V channel of HSV to [0,1] and count the proportion
    of pixels where V > 0.7. Higher scores indicate brighter, clearer images.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added 'clarity' column containing scores between 0 and 1
            where 1 means all pixels have V > 0.7
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filePath'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue
            
            # Convert to HSV and extract V channel (brightness)
            if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
                # For grayscale images, use the intensity values as V channel
                v_channel = image / 255.0  # Normalize to [0,1]
            else:  # Color image
                # Convert to HSV and extract V channel
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                v_channel = hsv_image[:, :, 2] / 255.0  # Normalize V channel to [0,1]

            # Calculate the proportion of pixels with V > 0.7
            clarity_mask = v_channel > 0.7
            clarity_score = np.sum(clarity_mask) / v_channel.size  # Proportion of high-brightness pixels
            
            df.loc[idx, 'clarity'] = clarity_score

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_clarity'] = error
            continue

    return df

def describe_blip(self, df_images): 
    """
    This function generates a textual description of an image using the BLIP model.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Clear GPU cache if available
    if self.cuda_availability:
        torch.cuda.empty_cache()

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters
        processor_path = self.config.get('describe_blip', {}).get('processor', 'Salesforce/blip-image-captioning-base')
        model_path = self.config.get('describe_blip', {}).get('model', 'Salesforce/blip-image-captioning-base')

        # Load processor and model
        try:
            processor = BlipProcessor.from_pretrained(processor_path)
            model = BlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
        except Exception as e:
            print(f"Error loading BLIP model: {str(e)}")
            return df

        # Initialize new columns with empty string
        df['descrBlip'] = ""

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_blip'] = "File not found"
                    continue

                # Load and process image
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_blip'] = f"Image load error: {str(e)}"
                    continue

                # Generate description
                try:
                    inputs = processor(image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs)
                    caption = processor.decode(outputs[0], skip_special_tokens=True)
                    df.loc[idx, 'descrBlip'] = caption
                except Exception as e:
                    error = f"Description generation error: {str(e)}"
                    print(f"Error generating description for {image_path}: {error}")
                    df.loc[idx, 'error_blip'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_blip'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in BLIP setup: {str(e)}")
        return df

def describe_llm(self, df_images, prompt="Describe the image."): 
    """
    This function generates a textual description of an image using a large language model (Phi-4-multimodal-instruct).
    The prompt to generate the description can be customized.
    Note: Requires large GPU VRAM (> 16GB).

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Define model path and prompts
        model_path = "microsoft/Phi-4-multimodal-instruct"
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>' 
        combined_prompt = f'{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}'

        # Load model and processor
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="cuda", 
                torch_dtype="auto", 
                trust_remote_code=True,
                _attn_implementation='eager'
            ).cuda()
            generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading LLM model: {str(e)}")
            return df

        # Initialize new columns with empty string
        df['descrLLM'] = ""

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_llm'] = "File not found"
                    continue

                # Load and process image
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_llm'] = f"Image load error: {str(e)}"
                    continue

                # Generate description
                try:
                    inputs = processor(text=combined_prompt, images=image, return_tensors='pt').to('cuda:0')
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=1000,
                        generation_config=generation_config,
                        num_logits_to_keep=1
                    )
                    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                    response = processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    df.loc[idx, 'descrLLM'] = response
                except Exception as e:
                    error = f"Description generation error: {str(e)}"
                    print(f"Error generating description for {image_path}: {error}")
                    df.loc[idx, 'error_llm'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_llm'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in LLM setup: {str(e)}")
        return df

def describe_llm_api(self, df_images, prompt="Describe the image."):
    """
    This function generates a textual description of an image using OpenAI's API (e.g., GPT-4 Vision).
    The prompt to generate the description can be customized.
    Requires an OpenAI API key to be set in the environment variable specified in the configuration.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :param prompt: The prompt to use for generating the description
    :return: DataFrame with added feature columns:
        - descrLLM_API: The generated description
        - error_llm_api: Error message if generation failed
    """

    import base64
    import os
    from openai import OpenAI

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters from config
        config_params = self.config.get('describe_llm_api', {}).get('parameters', {})
        model = config_params.get('model', 'gpt-4o')
        api_key_env_var = config_params.get('api_key_env_var', 'OPENAI_API_KEY')
        max_tokens = config_params.get('max_tokens', 1000)
        
        # Get API key from environment variable
        api_key = os.getenv(api_key_env_var)
        
        if not api_key:
            error_msg = f"OpenAI API key not found. Please set the environment variable '{api_key_env_var}'."
            print(f"Error: {error_msg}")
            df['error_llm_api'] = error_msg
            return df

        # Initialize OpenAI client and test connection
        try:
            client = OpenAI(api_key=api_key)
            # Test connection with a simple API call
            # We'll do a minimal test by checking if we can create a client
            # The actual connection test happens on first API call
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI client: {str(e)}"
            print(f"Error: {error_msg}")
            df['error_llm_api'] = error_msg
            return df

        # Initialize new columns
        df['descrLLM_API'] = ""
        df['error_llm_api'] = ""

        # Iterate over all images using enumerate on the DataFrame column
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_llm_api'] = "File not found"
                    continue

                # Load and encode image
                try:
                    with open(image_path, 'rb') as image_file:
                        image_data = image_file.read()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_llm_api'] = f"Image load error: {str(e)}"
                    continue

                # Generate description using OpenAI API
                try:
                    response = client.responses.create(
                        model=model,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                                ],
                            }
                        ],
                        max_tokens=max_tokens
                    )
                    
                    description = response.output_text
                    df.loc[idx, 'descrLLM_API'] = description
                    
                except Exception as e:
                    error = f"OpenAI API error: {str(e)}"
                    print(f"Error generating description for {image_path}: {error}")
                    df.loc[idx, 'error_llm_api'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_llm_api'] = error
                continue

        return df
    
    except Exception as e:
        print(f"Error in OpenAI API setup: {str(e)}")
        df['error_llm_api'] = f"Setup error: {str(e)}"
        return df

def detect_faces(self, df_images):
    '''
    This function:
    0. Takes in a DataFrame with a column 'filePath'
    1. For each image in the DataFrame, detect faces
    2. Returns a DataFrame with the faces detected
    ''' 

    def get_faces(self, path_img):
        '''
        This function:
        0. Takes in a path to an img
        1. extracts all faces (bb, landmarks) as a dict and returns it
        '''
        # Initialize detector
        if self.cuda_availability:
            detector = MTCNN(device="GPU:0")
        else:
            detector = MTCNN(device="CPU:0")

        # read image
        img = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
        
        # predict faces
        faces = detector.detect_faces(img, box_format="xywh", threshold_onet=0.85)

        # reformat datastructure
        faces_dict = {}
        if bool(faces):
            # iterate over list of dicts
            counter = 1
            for entry in faces:
                faces_dict["face_" + str(counter)] = {
                    "score": entry["confidence"],
                    "facial_area": entry["box"],
                    "landmarks": {
                        "right_eye": entry["keypoints"]["right_eye"],
                        "left_eye": entry["keypoints"]["left_eye"],
                        "nose": entry["keypoints"]["nose"],
                        "mouth_right": entry["keypoints"]["mouth_right"],
                        "mouth_left": entry["keypoints"]["mouth_left"]
                    }
                }
                counter += 1
        return faces_dict




    # Clear GPU cache if available
    if self.cuda_availability:
        torch.cuda.empty_cache()

    # Get confidence threshold from config
    confidence_threshold = self.config.get('features', {}).get('detect_faces', {}).get('parameters', {}).get('confidence_threshold', 0.9)
    
    # Initialize results DataFrame
    df_results = pd.DataFrame()
    df_results['filePath'] = df_images['filePath']
    
    # Process each image
    face_counts = []
    face_scores = []
    face_areas_abs = []
    face_areas_rel = []
    
    for idx, img_path in enumerate(tqdm(df_images['filePath'])):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            face_counts.append(0)
            face_scores.append(None)
            face_areas_abs.append(None)
            face_areas_rel.append(None)
            continue
            
        # Detect faces
        faces = get_faces(self, img_path)
        
        if faces:
            # Count faces
            face_counts.append(len(faces))
            
            # Calculate average confidence score
            scores = [face['score'] for face in faces.values()]
            face_scores.append(sum(scores) / len(scores))
            
            # Calculate total face area
            areas_abs = [(face['facial_area'][2] * face['facial_area'][3]) for face in faces.values()]
            face_areas_abs.append(sum(areas_abs))

            # Calculate relative face area
            areas_rel = [(face['facial_area'][2] * face['facial_area'][3] / (height * width)) for face in faces.values()]
            face_areas_rel.append(sum(areas_rel))
        else:
            face_counts.append(0)
            face_scores.append(None)
            face_areas_abs.append(0)
            face_areas_rel.append(0)
    
    # Add results to DataFrame
    df_results['face_count'] = face_counts
    df_results['face_confidence_avg'] = face_scores
    df_results['face_area_total_abs'] = face_areas_abs    
    df_results['face_area_total_rel'] = [min(1, area) if area is not None else None for area in face_areas_rel]
    
    return df_results

def detect_objects(self, df_images):
    """
    Detect objects in images using a pre-trained model.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added columns indicating the presence of detected objects
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters
        objects_to_detect = self.config.get("features", {}).get("detect_objects", {}).get("parameters", {}).get("objects_to_detect", [])
        print(f"Objects to detect: {objects_to_detect}")

        # Initialize model
        try:
            model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(self.device)
            processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return df

        # Initialize columns
        for obj in objects_to_detect:
            column_name = f"contains_{obj.lower()}"
            df[column_name] = False

        # Process images
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Load and process image
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Prepare inputs
                prompt = "<OD>"
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(self.device)

                # Generate predictions
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"] if "input_ids" in inputs else None,
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=4096,
                    num_beams=3,
                    do_sample=False
                )

                # Process results
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task="<OD>",
                    image_size=(image.width, image.height)
                )

                detected_objects = parsed_answer.get('<OD>', {}).get('labels', [])
                detected_objects_lower = [obj.lower() for obj in detected_objects]

                for obj in objects_to_detect:
                    column_name = f"contains_{obj.lower()}"
                    df.loc[idx, column_name] = obj.lower() in detected_objects_lower

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_object_detection'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in object detection setup: {str(e)}")
        return df

def estimate_noise(self, df_images):
    """
    Estimate the noise level in images using a Laplacian kernel convolution method.
    The function calculates a sigma value that represents the amount of noise,
    where values above 10 indicate significant noise presence in the image.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added 'noise' column containing the sigma values
            where higher values (>10) indicate noisier images
    :note: Uses grayscale conversion and 3x3 Laplacian kernel for noise estimation
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filePath'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image using cv2
            img = cv2.imread(image_path)
            if img is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            H, W = gray.shape

            M = [[1, -2, 1],
                 [-2, 4, -2],
                 [1, -2, 1]]

            sigma = np.sum(np.sum(np.absolute(convolve2d(gray, M))))
            sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

            # Value more than 10 indicates a noisy image
            df.loc[idx, 'noise'] = sigma

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_noise_detection'] = error
            continue

    return df

def extract_basic_image_features(self, df_images):
    """
    Extract basic features from the images and add them as columns to the DataFrame.
    Features include: dimensions (height, width), file size, color statistics (RGB means),
    HSV color space metrics, grayscale mean, and Shannon entropy.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added feature columns:
        - height: image height in pixels
        - width: image width in pixels
        - size_kb: file size in kilobytes
        - r_mean, g_mean, b_mean: average values for RGB channels
        - hueMean, saturationMean, brightness_mean: average values in HSV color space
        - greyscale_mean: average intensity in grayscale
        - shannon_entropy: measure of image complexity/information content
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filePath'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image using cv2
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue

            # Color space conversions
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # File specific features
            df.loc[idx, 'fileName'] = os.path.splitext(os.path.basename(image_path))[0]
            df.loc[idx, 'fileType'] = os.path.splitext(image_path)[1][1:].lower()
            df.loc[idx, 'fileSize'] = os.path.getsize(image_path) / 1024
            df.loc[idx, 'fileCreationTime'] = os.path.getctime(image_path)

            # # Image meta data
            # df.loc[idx, 'lens'] = image.get('Lens', 'NA')
            # df.loc[idx, 'focalLength'] = image.get('FocalLength', 'NA')
            # df.loc[idx, 'aperture'] = image.get('Aperture', 'NA')
            # df.loc[idx, 'exposureTime'] = image.get('ExposureTime', 'NA')
            # df.loc[idx, 'ISO'] = image.get('ISO', 'NA')
            # df.loc[idx, 'shutterSpeed'] = image.get('ShutterSpeedValue', 'NA')
            # df.loc[idx, 'whiteBalance'] = image.get('WhiteBalance', 'NA')
            # df.loc[idx, 'flash'] = image.get('Flash', 'NA')
            # df.loc[idx, 'meteringMode'] = image.get('MeteringMode', 'NA')
            # df.loc[idx, 'exposureProgram'] = image.get('ExposureProgram', 'NA')

            # Image dimensions
            df.loc[idx, 'height'] = image.shape[0]
            df.loc[idx, 'width'] = image.shape[1]
            df.loc[idx, 'aspectRatio'] = df.loc[idx, 'width'] / df.loc[idx, 'height']

            # Color channels
            df.loc[idx, 'rMean'] = np.mean(image_rgb[:, :, 0])
            df.loc[idx, 'rStd'] = np.std(image_rgb[:, :, 0])
            df.loc[idx, 'gMean'] = np.mean(image_rgb[:, :, 1])
            df.loc[idx, 'gStd'] = np.std(image_rgb[:, :, 1])
            df.loc[idx, 'bMean'] = np.mean(image_rgb[:, :, 2])
            df.loc[idx, 'bStd'] = np.std(image_rgb[:, :, 2])

            # HSV channels
            df.loc[idx, 'hueMean'] = np.mean(image_hsv[:, :, 0])
            df.loc[idx, 'hueStd'] = np.std(image_hsv[:, :, 0])
            # df.loc[idx, 'saturationMean'] = np.mean(image_hsv[:, :, 1])
            # df.loc[idx, 'saturationStd'] = np.std(image_hsv[:, :, 1])
            # df.loc[idx, 'brightnessMean'] = np.mean(image_hsv[:, :, 2])
            # df.loc[idx, 'brightnessStd'] = np.std(image_hsv[:, :, 2])

            # # Grayscale
            df.loc[idx, 'greyscaleMean'] = np.mean(image_gray)
            df.loc[idx, 'greyscaleStd'] = np.std(image_gray)

            # # Shannon entropy
            df.loc[idx, 'shannonEntropy'] = measure.shannon_entropy(image_gray)

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_basic_img_features'] = error
            continue

    return df

def extract_blur_value(self, df_images):
    """
    Calculate the blur value for each image using Laplacian variance.
    A lower value (< 100) indicates a blurry image, while higher values indicate sharper images.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added 'blur' column containing the Laplacian variance score
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filePath'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image using cv2
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Set threshold at 100. Value below 100 indicates a blurry image
            df.loc[idx, 'blur'] = cv2.Laplacian(gray, cv2.CV_64F).var()

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_blur_detection'] = error
            continue

    return df

def felzenszwalb_segmentation(self, df_images): 
    """
    This function performs Felzenszwalb segmentation on the images in the input DataFrame.
    The function adds a new column to the DataFrame with the segmented images in RGB format.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters from config if available
        scale = self.config.get("felzenszwalb_segmentation", {}).get("scale", 100)
        sigma = self.config.get("felzenszwalb_segmentation", {}).get("sigma", 0.5)
        min_size = self.config.get("felzenszwalb_segmentation", {}).get("min_size", 50)

        # Initialize new columns
        df['felzenszwalbSegmentationRGB'] = None

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_segmentation'] = "File not found"
                    continue

                # Load image
                try:
                    image = imread(image_path)
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_segmentation'] = f"Image load error: {str(e)}"
                    continue

                # Perform segmentation
                try:
                    segments = felzenszwalb(
                        image, 
                        scale=scale, 
                        sigma=sigma, 
                        min_size=min_size
                    )
                    segmented_image = label2rgb(segments, image, kind='avg')
                    df.loc[idx, 'felzenszwalbSegmentationRGB'] = segmented_image

                except Exception as e:
                    error = f"Segmentation error: {str(e)}"
                    print(f"Error performing segmentation for {image_path}: {error}")
                    df.loc[idx, 'error_segmentation'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_segmentation'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in segmentation setup: {str(e)}")
        return df

def get_color_features(self, df_images):
    """
    Computes color-related features for images, including:
      - brightness: Mean brightness, normalized to [0,1]
      - saturation: Mean saturation, normalized to [0,1]
      - contrast: Brightness contrast, defined as the standard deviation of the V channel in HSV, normalized to [0,1]
      - colorfulness: Colorfulness score based on Hasler and Suesstrunk (2003), normalized to [0,1]

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Get parameters
    color_config = self.config.get("get_color_features", {})
    
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Initialize new columns with NaN
    df['brightnessMean'] = np.nan
    df['saturationMean'] = np.nan
    df['contrast'] = np.nan
    df['colorfulness'] = np.nan

    # Iterate over all image paths
    for idx, image_path in enumerate(tqdm(df_images['filePath'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue

            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)

            # Compute features
            brightness_val = np.mean(V / 255.0)
            saturation_val = np.mean(S / 255.0)
            contrast_val = np.std(V / 255.0) * 2

            # Compute colorfulness
            B = image[:, :, 0].astype("float")
            G = image[:, :, 1].astype("float")
            R = image[:, :, 2].astype("float")
            rg = R - G
            yb = 0.5 * (R + G) - B
            std_rg = np.std(rg)
            std_yb = np.std(yb)
            mean_rg = np.mean(rg)
            mean_yb = np.mean(yb)
            colorfulness_raw = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)

            # Store computed features
            df.loc[idx, 'brightnessMean'] = brightness_val
            df.loc[idx, 'saturationMean'] = saturation_val
            df.loc[idx, 'contrast'] = contrast_val
            df.loc[idx, 'colorfulness'] = colorfulness_raw

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_color_features'] = error
            continue

    try:
        # Normalize colorfulness across all images to [0,1]
        raw_colorfulness = df['colorfulness'].astype('float').values
        min_c = np.nanmin(raw_colorfulness)
        max_c = np.nanmax(raw_colorfulness)
        if max_c - min_c == 0:
            df['colorfulness'] = 0.0
        else:
            df['colorfulness'] = (raw_colorfulness - min_c) / (max_c - min_c)
    except Exception as e:
        print(f"Error normalizing colorfulness: {str(e)}")

    return df

def get_composition_features(self, df_images):
    """
    Computes composition features for each image according to the specifications.

    For each image, the following features are computed:
      - diagonalDominance: The normalized (inverted) minimum distance from the salient center to the two image diagonals.
      - ruleOfThirds: The normalized (inverted) minimum distance from the salient center to the four intersections of a 3×3 grid.
      - physicalVisualBalance:
            • physicalVisualBalance_vertical: 1 minus the normalized vertical distance between the salient center and the image center.
            • physicalVisualBalance_horizontal: 1 minus the normalized horizontal distance between the salient center and the image center.
      - colorVisualBalance:
            • colorVisualBalance_vertical: 1 minus the normalized average Euclidean color distance between top and bottom symmetric pixels.
            • colorVisualBalance_horizontal: 1 minus the normalized average Euclidean color distance between left and right symmetric pixels.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files.
    :return: DataFrame with added feature columns:
             - diagonalDominance
             - ruleOfThirds
             - physicalVisualBalance_vertical, physicalVisualBalance_horizontal
             - colorVisualBalance_vertical, colorVisualBalance_horizontal
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Initialize composition feature columns
        df['diagonalDominance'] = np.nan
        df['ruleOfThirds'] = np.nan
        df['physicalVisualBalanceVertical'] = np.nan
        df['physicalVisualBalanceHorizontal'] = np.nan
        df['colorVisualBalanceVertical'] = np.nan
        df['colorVisualBalanceHorizontal'] = np.nan

        # Create a saliency detector using OpenCV's StaticSaliencySpectralResidual
        saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

        # Maximum possible color difference in RGB space (Euclidean distance)
        max_color_distance = np.sqrt(255 ** 2 + 255 ** 2 + 255 ** 2)  # ≈441.67

        # Iterate over each image in the DataFrame
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Load the image using cv2
                image = cv2.imread(image_path)
                if image is None:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Get image dimensions (height and width)
                H, W = image.shape[:2]

                # Compute saliency map
                success, saliencyMap = saliency_detector.computeSaliency(image)
                if not success or saliencyMap is None:
                    print(f"Warning: Failed to compute saliency map for {image_path}")
                    saliencyMap = np.ones((H, W), dtype="float32")
                else:
                    saliencyMap = saliencyMap.squeeze()
                    if saliencyMap.ndim != 2:
                        saliencyMap = saliencyMap[:, :, 0]

                # After computing saliency map:
                # 1. Apply superpixel segmentation (e.g., using SLIC or similar)
                segments = slic(image, n_segments=10, compactness=10)

                # 2. Calculate average saliency for each segment, excluding segments with no pixels (i.e., skip)
                segment_saliencies = []
                segment_ids = []  # Store the actual segment IDs
                for segment_id in np.unique(segments):
                    segment_mask = (segments == segment_id)
                    if np.count_nonzero(segment_mask) == 0:
                        avg_saliency = np.nan
                    else:
                        avg_saliency = np.mean(saliencyMap[segment_mask])
                    segment_saliencies.append(avg_saliency)
                    segment_ids.append(segment_id)  # Store the actual segment ID

                # 3. Find salient region (segment with highest average saliency)
                max_saliency_idx = np.argmax(segment_saliencies)  # Index in the list
                salient_segment_id = segment_ids[max_saliency_idx]  # Actual segment ID
                salient_mask = (segments == salient_segment_id)

                # 4. Compute salient region center
                salient_center_y, salient_center_x = np.mean(np.where(salient_mask), axis=1)

                # 5. Use salient_center_x, salient_center_y for composition calculations

                # 1. Diagonal Dominance
                d1 = np.abs(H * salient_center_x - W * salient_center_y) / np.sqrt(W ** 2 + H ** 2)  # Distance to diagonal from top-left to bottom-right
                d2 = np.abs(H * salient_center_x + W * salient_center_y - H * W) / np.sqrt(
                    W ** 2 + H ** 2)  # Distance to diagonal from top-right to bottom-left
                d_min = min(d1, d2)
                D_max_diag = (W*H) / np.sqrt(W**2 + H**2)
                diagonalDominance = 1.0 - (d_min / D_max_diag)
                diagonalDominance = float(np.clip(diagonalDominance, 0.0, 1.0))

                # 2. Rule of Thirds
                intersections = [(W / 3.0, H / 3.0), (2 * W / 3.0, H / 3.0),
                                 (W / 3.0, 2 * H / 3.0), (2 * W / 3.0, 2 * H / 3.0)]
                min_distance = min(np.hypot(salient_center_x - x, salient_center_y - y)
                   for (x,y) in intersections)
                D_max_rot = np.sqrt(W**2 + H**2) / 3.0
                ruleOfThirds = 1.0 - (min_distance / D_max_rot)
                ruleOfThirds = float(np.clip(ruleOfThirds, 0.0, 1.0))

                # 3. Physical Visual Balance
                # Calculate weighted center of the photo by weighting segment centers by their saliency
                # Vectorized approach for efficiency

                # Create coordinate grids
                Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

                # Calculate weighted center using vectorized operations
                total_weight = np.sum(saliencyMap)
                if total_weight > 0:
                    cx = np.sum(X * saliencyMap) / total_weight
                    cy = np.sum(Y * saliencyMap) / total_weight
                else:
                    cx, cy = W / 2.0, H / 2.0  # Fallback to image center

                # Calculate physical visual balance scores
                vertical_balance = (1.0 - np.abs(cy - H/2.0) / (H/2.0)) if H > 0 else 0
                vertical_balance = np.clip(vertical_balance, 0.0, 1.0)

                horizontal_balance = (1.0 - np.abs(cx - W/2.0) / (W/2.0)) if W > 0 else 0
                horizontal_balance = np.clip(horizontal_balance, 0.0, 1.0)

                # 4. Color Visual Balance ----
                max_color_distance = np.sqrt(3*(255**2))  # ~441.67
                # vertical
                h_half = H // 2
                if h_half > 0:
                    top_half    = image[:h_half].astype(np.float32)
                    bottom_half = image[H - h_half:][::-1].astype(np.float32)
                    avg_diff_v = np.mean(np.linalg.norm(top_half - bottom_half, axis=2))
                    vertical_color_balance = 1.0 - (avg_diff_v / max_color_distance)
                    vertical_color_balance = float(np.clip(vertical_color_balance, 0.0, 1.0))
                else:
                    vertical_color_balance = 1.0
                # horizontal
                w_half = W // 2
                if w_half > 0:
                    left_half  = image[:, :w_half].astype(np.float32)
                    right_half = image[:, W - w_half:][:, ::-1].astype(np.float32)
                    avg_diff_h = np.mean(np.linalg.norm(left_half - right_half, axis=2))
                    horizontal_color_balance = 1.0 - (avg_diff_h / max_color_distance)
                    horizontal_color_balance = float(np.clip(horizontal_color_balance, 0.0, 1.0))
                else:
                    horizontal_color_balance = 1.0

                # Save features into DataFrame
                df.loc[idx, 'diagonalDominance'] = diagonalDominance
                df.loc[idx, 'ruleOfThirds'] = ruleOfThirds
                df.loc[idx, 'physicalVisualBalanceVertical'] = vertical_balance
                df.loc[idx, 'physicalVisualBalanceHorizontal'] = horizontal_balance
                df.loc[idx, 'colorVisualBalanceVertical'] = vertical_color_balance
                df.loc[idx, 'colorVisualBalanceHorizontal'] = horizontal_color_balance

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_composition'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in composition feature extraction setup: {str(e)}")
        return df

def get_figure_ground_relationship_features(self, df_images):
    """
    Computes figure-ground relationship features for each image according to the specifications.
    
    For each image, the following features are computed:
      - sizeDifference: The absolute difference between the number of figure pixels and background pixels,
                        normalized by the total number of pixels.
      - colorDifference: The Euclidean distance between the average RGB vector of the figure and that of the background,
                         normalized to [0, 1] (using 441.67 as the maximum difference).
      - textureDifference: The absolute difference between the edge densities (using Canny) of the figure and background,
                           normalized to [0, 1].
      - depthOfField: Computed for each HSV dimension (hue, saturation, value) as follows:
            • The image is divided into 16 equal regions.
            • For each HSV channel, the high-frequency (detail) coefficients are computed using a Daubechies wavelet (pywt.dwt2).
            • The score is defined as the sum of absolute detail coefficients in the center four regions divided by the sum
              of absolute detail coefficients over all 16 regions.
            A higher score indicates a lower depth of field.
    
    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files.
    :return: DataFrame with added feature columns:
             - sizeDifference
             - colorDifference
             - textureDifference
             - depthOfFieldHue, depthOfFieldSaturation, depthOfFieldValue
    """

    # Get parameters
    fg_config = self.config.get("get_figure_ground_relationship_features", {})
    saliency_threshold = fg_config.get("saliency_threshold", 0.5)
    canny_edge_low_threshold = fg_config.get("canny_edge_low_threshold", 100)
    canny_edge_high_threshold = fg_config.get("canny_edge_high_threshold", 200)
    
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    # Initialize feature columns using camelCase naming
    df['sizeDifference'] = np.nan
    df['colorDifference'] = np.nan
    df['textureDifference'] = np.nan
    df['depthOfFieldHue'] = np.nan
    df['depthOfFieldSaturation'] = np.nan
    df['depthOfFieldValue'] = np.nan

    # Create a saliency detector for figure-ground segmentation
    saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    
    # Maximum possible color difference in RGB (Euclidean distance)
    max_color_distance = np.sqrt(255**2 + 255**2 + 255**2)  # ≈441.67

    for idx, image_path in enumerate(tqdm(df_images['filePath'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue

            H, W = image.shape[:2]
            total_pixels = H * W
            
            # Compute saliency map and threshold to segment figure (salient) vs background
            success, saliencyMap = saliency_detector.computeSaliency(image)
            if not success or saliencyMap is None:
                print(f"Warning: Failed to compute saliency map for {image_path}")
                saliencyMap = np.ones((H, W), dtype="float32")
            else:
                saliencyMap = saliencyMap.squeeze()
                if saliencyMap.ndim != 2:
                    saliencyMap = saliencyMap[:, :, 0]
            # Normalize saliency map to [0,1] if not already
            saliencyMap = cv2.normalize(saliencyMap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            _, figureMask = cv2.threshold(saliencyMap, saliency_threshold, 1, cv2.THRESH_BINARY)
            figureMask = figureMask.astype(np.uint8)
            backgroundMask = 1 - figureMask
            
            # 1. Size Difference
            figure_pixels = np.sum(figureMask, dtype=np.int64)
            background_pixels = np.sum(backgroundMask, dtype=np.int64)
            sizeDifference = np.abs(figure_pixels - background_pixels) / float(total_pixels)
            
            # 2. Color Difference
            # Compute average RGB for figure and background; if a region is empty, use zeros.
            figure_pixels_indices = np.where(figureMask == 1)
            background_pixels_indices = np.where(backgroundMask == 1)
            if figure_pixels > 0:
                avgRGB_figure = np.mean(image[figure_pixels_indices], axis=0)
            else:
                avgRGB_figure = np.zeros(3)
            if background_pixels > 0:
                avgRGB_background = np.mean(image[background_pixels_indices], axis=0)
            else:
                avgRGB_background = np.zeros(3)
            color_diff = np.linalg.norm(avgRGB_figure - avgRGB_background)
            colorDifference = np.clip(color_diff / max_color_distance, 0, 1)
            
            # 3. Texture Difference
            # Use Canny edge detection on grayscale image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, canny_edge_low_threshold, canny_edge_high_threshold)
            # Compute edge density for figure and background
            if figure_pixels > 0:
                edge_density_figure = np.sum(edges[figure_pixels_indices] > 0) / float(figure_pixels)
            else:
                edge_density_figure = 0
            if background_pixels > 0:
                edge_density_background = np.sum(edges[background_pixels_indices] > 0) / float(background_pixels)
            else:
                edge_density_background = 0
            textureDifference = np.clip(np.abs(edge_density_figure - edge_density_background), 0, 1)
            
            # 4. Depth of Field (for each HSV channel)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
            # Divide the image into a 4x4 grid (16 regions)
            grid_rows, grid_cols = 4, 4
            region_h = H // grid_rows
            region_w = W // grid_cols
            
            # Define indices for center 4 regions (positions (1,1), (1,2), (2,1), (2,2) in 0-indexing)
            center_regions = [(1,1), (1,2), (2,1), (2,2)]
            
            # Initialize sums for each channel: hue, saturation, value
            total_detail = {'hue': 0, 'sat': 0, 'val': 0}
            center_detail = {'hue': 0, 'sat': 0, 'val': 0}
            
            # For each region in the 4x4 grid
            for i in range(grid_rows):
                for j in range(grid_cols):
                    y0 = i * region_h
                    x0 = j * region_w
                    # Make sure to include remaining pixels for the last row/column
                    y1 = H if i == grid_rows - 1 else (i+1)*region_h
                    x1 = W if j == grid_cols - 1 else (j+1)*region_w
                    
                    region = hsv[y0:y1, x0:x1, :]
                    # For each channel, compute the sum of absolute high-frequency coefficients using a Daubechies wavelet.
                    # We perform a single-level 2D DWT.
                    for idx_channel, key in enumerate(['hue', 'sat', 'val']):
                        coeffs2 = pywt.dwt2(region[:, :, idx_channel], 'db1')
                        # coeffs2 returns (LL, (LH, HL, HH)); we sum absolute values of high-frequency coefficients.
                        (_, (LH, HL, HH)) = coeffs2
                        detail_sum = np.sum(np.abs(LH)) + np.sum(np.abs(HL)) + np.sum(np.abs(HH))
                        total_detail[key] += detail_sum
                        if (i, j) in center_regions:
                            center_detail[key] += detail_sum
            
            # For each channel, compute depth-of-field score as center_detail/total_detail.
            # Avoid division by zero.
            dof_hue = center_detail['hue'] / total_detail['hue'] if total_detail['hue'] != 0 else 0
            dof_sat = center_detail['sat'] / total_detail['sat'] if total_detail['sat'] != 0 else 0
            dof_val = center_detail['val'] / total_detail['val'] if total_detail['val'] != 0 else 0

            # Save features into DataFrame (clipped to [0, 1] where applicable)
            df.loc[idx, 'sizeDifference'] = sizeDifference
            df.loc[idx, 'colorDifference'] = colorDifference
            df.loc[idx, 'textureDifference'] = textureDifference
            df.loc[idx, 'depthOfFieldHue'] = np.clip(dof_hue, 0, 1)
            df.loc[idx, 'depthOfFieldSaturation'] = np.clip(dof_sat, 0, 1)
            df.loc[idx, 'depthOfFieldValue'] = np.clip(dof_val, 0, 1)
            
        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_figure_ground'] = error
            continue

    return df

def get_ocr_text(self, df_images):
    """
    Extract text from images using pytesseract and identify the language.
    
    :param self: IA object
    :param df_images: DataFrame containing image filePaths
    :return: DataFrame with OCR results (ocrHasText, ocrText, and ocrLanguage columns)
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Initialize tesseract for Windows
        if platform.system() == "Windows":
            try:
                tesseract_path = self.config.get("features", {}).get("get_ocr_text", {}).get("parameters", {}).get("windows_path_to_tesseract")
                if tesseract_path:
                    print(f"Windows system detected. Using Tesseract path: {tesseract_path}")
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
            except Exception as e:
                error = f"Error configuring Tesseract: {str(e)}"
                print(error)
                traceback.print_exc()
                return df

        # Initialize new columns
        df['ocrHasText'] = False
        df['ocrText'] = ""
        df['ocrLanguage'] = ""
        
        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_ocr'] = "File not found"
                    continue

                # Open the image with PIL
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_ocr'] = f"Image load error: {str(e)}"
                    continue

                # Perform OCR using pytesseract
                try:
                    text = pytesseract.image_to_string(image).strip()
                    has_text = len(text) > 0
                    
                    # Store OCR text result
                    df.loc[idx, 'ocrHasText'] = has_text
                    df.loc[idx, 'ocrText'] = text

                    # Detect language if text is present
                    if has_text:
                        try:
                            language = detect(text)
                        except LangDetectException:
                            language = "unknown"
                        except Exception as e:
                            language = "error"
                            df.loc[idx, 'error_ocr_lang'] = f"Language detection error: {str(e)}"
                    else:
                        language = ""

                    # Store language result
                    df.loc[idx, 'ocrLanguage'] = language

                except Exception as e:
                    error = f"OCR processing error: {str(e)}"
                    print(f"Error processing OCR for {image_path}: {error}")
                    df.loc[idx, 'error_ocr'] = error
                    df.loc[idx, 'ocrHasText'] = False
                    df.loc[idx, 'ocrText'] = ""
                    df.loc[idx, 'ocrLanguage'] = ""
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_ocr'] = error
                df.loc[idx, 'ocrHasText'] = False
                df.loc[idx, 'ocrText'] = ""
                df.loc[idx, 'ocrLanguage'] = ""
                continue

        return df

    except Exception as e:
        print(f"Error in OCR setup: {str(e)}")
        return df

def predict_imagenet_classes_yolo11(self, df_images):
    """
    Predicts ImageNet classes in a list of images using YOLO11 classification model.

    :param self: IA object
    :param df_images: DataFrame containing image filePaths.
    :return: A DataFrame containing ImageNet labels and their prediction probabilities for each image.
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    # Dictionary to collect all class probabilities before creating DataFrame
    all_probs = {}
    
    try:
        # Check if weights are downloaded already, otherwise download them
        download_weights(
            weight_filename='yolo11n-cls.pt', 
            weight_url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt'
        )

        # Load the YOLO11 classification model
        model = YOLO('../image_analyzer/weights/yolo11n-cls.pt').to(self.device)
        
        # Initialize progress bar
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Load image
                img =   cv2.imread(image_path)
                if img is None:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Perform inference with the classification model
                results = model(img, verbose=False)
                
                # Process classification results
                for result in results:
                    # Get the probs attribute which contains probabilities for all classes
                    probs = result.probs
                    
                    # Add all class probabilities to our temporary dictionary
                    for i, prob in enumerate(probs.data.tolist()):
                        class_name = result.names[i]
                        column_name = f"imagenet_{class_name}"
                        
                        if column_name not in all_probs:
                            all_probs[column_name] = [0.0] * len(df_images)
                        
                        all_probs[column_name][idx] = prob

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_yolo11_imagenet'] = error
                continue
        
        # Create a DataFrame from the collected probabilities and join with original DataFrame
        probs_df = pd.DataFrame(all_probs)
        result_df = pd.concat([df, probs_df], axis=1)
        return result_df

    except Exception as e:
        print(f"Error in YOLO11 ImageNet classification setup: {str(e)}")
        return df

def predict_coco_labels_yolo11(self, df_images):
    """
    Predicts COCO labels in a list of images.

    :param self: IA object
    :param df_images: DataFrame containing image filePaths.
    :return: A DataFrame containing COCO labels and their prediction probabilities for each image.
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Check if weights are downloaded already, otherwise download them
        download_weights(
            weight_filename='yolov11n.pt', 
            weight_url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
        )
        download_weights(
            weight_filename='coco.names', 
            weight_url='https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names'
        )

        # Load the YOLOv11 model
        model = YOLO('../image_analyzer/weights/yolov11n.pt')

        # Load COCO labels
        try:
            with open('../image_analyzer/weights/coco.names', 'r') as f:
                classes = ['coco_' + line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading COCO labels: {str(e)}")
            return df

        # Initialize columns for each class
        for label in classes:
            df[label] = 0.0

        # Iterate over all images
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Load image
                img =   cv2.imread(image_path)
                if img is None:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Perform inference
                results = model(img, verbose=False)

                # Analyze the outputs
                for result in results:
                    for detection in result.boxes:
                        class_id = int(detection.cls)
                        confidence = float(detection.conf)
                        if confidence > df.at[idx, classes[class_id]]:
                            df.at[idx, classes[class_id]] = confidence

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_yolo11_coco'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in YOLO11 COCO detection setup: {str(e)}")
        return df

def self_similarity(self, df_images):
    """
    This function calculates the self-similarity of each image using the power spectrum of the Fourier transform.
    The closer the slope of the power spectrum to -2, the more self-similar the image is.
    The slope is mapped to a similarity score between 0 and 1 using a Gaussian function.
    A score of 1 indicates perfect self-similarity and a score of 0 indicates no self-similarity.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Initialize column
        df['selfSimilarity'] = np.nan

        # Define linear fit function for curve fitting
        def linear_fit(x, a, b):
            return a * x + b

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_similarity'] = "File not found"
                    continue

                # Load image
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                        df.loc[idx, 'error_similarity'] = "Image load failed"
                        continue
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_similarity'] = f"Image load error: {str(e)}"
                    continue

                # Calculate self-similarity
                try:
                    # Compute Fourier transform and power spectrum
                    f_transform = np.fft.fft2(img)
                    f_transform_shifted = np.fft.fftshift(f_transform)
                    power_spectrum = np.abs(f_transform_shifted) ** 2

                    # Calculate radial profile
                    h, w = power_spectrum.shape
                    y, x = np.indices((h, w))
                    center = (h // 2, w // 2)
                    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(np.int32)
                    radial_mean = np.bincount(r.ravel(), weights=power_spectrum.ravel()) / np.bincount(r.ravel())

                    # Prepare data for fitting
                    valid = (radial_mean > 0) & (np.arange(len(radial_mean)) > 0)
                    freqs = np.arange(len(radial_mean))[valid]
                    power = radial_mean[valid]
                    log_freqs = np.log(freqs)
                    log_power = np.log(power)

                    # Fit curve and calculate similarity score
                    slope, intercept = curve_fit(linear_fit, log_freqs, log_power)[0]
                    similarity_score = np.exp(-0.5 * abs(slope + 2))

                    df.loc[idx, 'selfSimilarity'] = similarity_score

                except Exception as e:
                    error = f"Similarity calculation error: {str(e)}"
                    print(f"Error calculating similarity for {image_path}: {error}")
                    df.loc[idx, 'error_similarity'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_similarity'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in self-similarity setup: {str(e)}")
        return df

def visual_complexity(self, df_images):
    """
    Calculate the visual complexity of each image by counting the number of regions in the binary image.

    :param self: IA object
    :param df_images: DataFrame containing a 'filePath' column with paths to image files
    :return: DataFrame with added 'visualComplexity' column containing the number of regions
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters from config
        threshold = self.config.get('visual_complexity', {}).get('threshold', 25000)
        
        # Initialize column
        df['visualComplexity'] = np.nan
        
        for idx, input_image_path in enumerate(tqdm(df_images['filePath'])):
            try:
                # Check if file exists
                if not os.path.exists(input_image_path):
                    print(f"Warning: File not found: {input_image_path}")
                    df.loc[idx, 'error_complexity'] = "File not found"
                    continue

                # Load image
                try:
                    img = io.imread(input_image_path)
                except Exception as e:
                    print(f"Warning: Failed to load image: {input_image_path}")
                    df.loc[idx, 'error_complexity'] = f"Image load error: {str(e)}"
                    continue

                # Process image
                try:
                    # Convert to grayscale if needed
                    if len(img.shape) > 2:
                        img = np.mean(img, axis=2).astype(np.uint8)

                    # Calculate adaptive threshold
                    thresh = filters.threshold_local(img, block_size=35)
                    binary_img = img > thresh
                    rp_tot = binary_img.shape[0] * binary_img.shape[1]
                    labeled_img = label(binary_img)
                    regions = regionprops(labeled_img)
                    threshold_val = rp_tot / threshold
                    r_spt = sum(1 for region in regions if region.area > threshold_val)

                    df.loc[idx, 'visualComplexity'] = r_spt

                except Exception as e:
                    error = f"Complexity calculation error: {str(e)}"
                    print(f"Error calculating complexity for {input_image_path}: {error}")
                    df.loc[idx, 'error_complexity'] = error
                    continue

            except Exception as e:
                error = f"Error processing {input_image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_complexity'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in visual complexity setup: {str(e)}")
        return df
