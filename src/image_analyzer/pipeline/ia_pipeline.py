# General imports
from fpdf import FPDF
import gc  
import os
import pandas as pd
import time
import torch
import random

# Project-specific imports
from ..utils.feature_extraction import *
from ..utils.helper_functions import load_config

class IA:
    """
    Image Analyzer pipeline class.

    This class encapsulates all functionalities needed to process images, 
    extract features, and manage the image analysis pipeline.
    """

    def __init__(self, config_path):
        """
        Initialize the Image Analyzer pipeline with a configuration dictionary.

        :param config_path: The path to the configuration file.
        """
        random.seed(0)

        # Store the config path for later saving
        self.config_path = config_path
        
        # Get parameters
        self.config = load_config(config_path)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(f'Timestamp: {self.timestamp}')
        self.input_dir = self.config.get("general", {}).get("input_dir")
        self.output_dir = self.config.get("general", {}).get("output_dir")
        self.verbose = self.config.get("general", {}).get("verbose", True)
        self.debug_mode = self.config.get("general", {}).get("debug_mode")
        self.debug_img_cnt = self.config.get("general", {}).get("debug_image_count")
        self.incl_summary_stats = self.config.get("general", {}).get("summary_stats", {}).get("active", True)
        self.multi_processing = self.config.get("general", {}).get("multi_processing", {}).get("active", False)
        self.num_processes = self.config.get("general", {}).get("multi_processing", {}).get("num_processes", 4)

        self.cuda_availability = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_availability else "cpu"
        # Console outputs
        if self.cuda_availability:
            print(f"### Using GPU (CUDA) ###")
        else:
            print(f"### Using CPU ###")

    def reset_pipeline(self):
        """
        Reset the pipeline state for a new batch.
        This method clears any cached data and resets internal state.
        """
        # Clear GPU cache if available
        if self.cuda_availability:
            torch.cuda.empty_cache()
        
        # Force garbage collection multiple times to ensure cleanup
        for _ in range(3):
            gc.collect()
        
        # Reset timestamp for new batch
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Update output directory with new timestamp
        if hasattr(self, 'config') and self.config:
            base_output_dir = self.config.get("general", {}).get("output_dir", "outputs")
            self.output_dir = os.path.join(base_output_dir, f"Image-Analyzer_run_{self.timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"### Pipeline reset for new batch - Timestamp: {self.timestamp} ###")

    def save_config_to_output(self, output_dir=None):
        """
        Save the configuration file used for this run to the output directory.
        This ensures reproducibility by preserving the exact configuration used.

        :param output_dir: The directory to save the config to. If not provided,
                          uses the current output directory.
        """
        if output_dir is None:
            output_dir = self.output_dir

        try:
            import yaml
            import shutil
            
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine the source config file path
            if self.config_path and os.path.exists(self.config_path):
                # If config_path exists, copy it directly
                config_source = self.config_path
                config_filename = f"configuration.yaml"
            else:
                # If config_path doesn't exist (e.g., when using built-in defaults),
                # save the current config dictionary as YAML
                config_source = None
                config_filename = f"configuration.yaml"
            
            config_dest = os.path.join(output_dir, config_filename)
            
            if config_source and os.path.exists(config_source):
                # Copy the original config file
                shutil.copy2(config_source, config_dest)
                print(f"### Configuration file copied to: {config_dest} ###")
            else:
                # Save the current config dictionary as YAML
                with open(config_dest, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                print(f"### Configuration saved to: {config_dest} ###")
                
        except Exception as e:
            print(f"### Warning: Could not save configuration file: {str(e)} ###")

    def process_batch(self, progress_callback=None):
        """
        Process a batch of images from a list of file paths.

        :param self: IA object
        :param progress_callback: Callback function to update progress with dict containing:
                                 - percentage: float (0-1)
                                 - current_function: str
                                 - status: str
                                 - message: str
        :return: A DataFrame containing the features extracted from each image.
        """
        # Create output directory
        self.output_dir = os.path.join(self.output_dir , f"Image-Analyzer_run_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the configuration file used for this run
        self.save_config_to_output()

        # Collect all images from input_dir
        image_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.ppm', '.pgm', '.pbm', '.gif', '.hdr', '.exr'))]
        if self.debug_mode and len(image_files) > self.debug_img_cnt:
            image_files = random.sample(image_files, self.debug_img_cnt)
        image_files.sort()

        # Initialize dataframe with all image files and ensure filePath is string type
        df_images = pd.DataFrame({'filePath': image_files})
        df_images['filePath'] = df_images['filePath'].astype(str)
        df_out = df_images.copy(deep=True)

        # Create a dataframe of feature extractor functions with active status and time tracking
        df_logs = pd.DataFrame({
            'functions': [
                extract_basic_image_features,
                get_color_features,
                calculate_hue_proportions,
                calculate_image_clarity,
                self_similarity,
                felzenszwalb_segmentation, 
                extract_blur_value,
                estimate_noise,
                calculate_aesthetic_scores,
                get_composition_features,
                get_figure_ground_relationship_features,
                visual_complexity,
                detect_objects,
                detect_faces,
                get_ocr_text,
                describe_blip,
                describe_llm,
                describe_llm_api,
                predict_coco_labels_yolo11,
                predict_imagenet_classes_yolo11
            ]
        })
        df_logs['active'] = None
        df_logs['seconds_needed'] = None

        # Save interim df_out and df_logs
        self.save_results(df_out)
        self.save_logs(df_logs)

        # Identify which functions should be processed
        active_functions = []
        for idx, row in df_logs.iterrows():
            func_name = row['functions'].__name__
            df_logs.at[idx, 'active'] = self.config.get('features', {}).get(func_name, {}).get('active', False)
            if df_logs.at[idx, 'active']:
                active_functions.append(func_name)

        if progress_callback:
            progress_callback({
                'percentage': 0.0,
                'current_function': 'Initializing',
                'status': 'starting',
                'message': f'Starting processing of {len(df_images)} images in {len(active_functions)} steps'
            })

        print(f"### Starting batch of n={len(df_images)} images ###")
        
        # Iterate over each function in the feature_extractors_df dataframe
        processed_count = 0
        for idx, row in df_logs.iterrows():
            # Flush cache
            gc.collect()
            if self.cuda_availability:
                torch.cuda.empty_cache()

            func = row['functions']
            func_name = func.__name__
            
            if row['active']:
                if progress_callback:
                    progress_callback({
                        'percentage': (processed_count+1) / len(active_functions),
                        'current_function': func_name,
                        'status': 'processing',
                        'message': f'Step {processed_count+1}/{len(active_functions)}: {func_name}'
                    })
                
                print("--------------------------------")
                print(f"Step {processed_count+1}/{len(active_functions)}: {func_name}()")

                tic = time.perf_counter()
                
                # Execute function
                df_temp = func(self, df_images)
                
                # Ensure filePath column is string type in temporary results
                df_temp['filePath'] = df_temp['filePath'].astype(str)
                
                # Append results to dataframe
                df_out = df_out.merge(df_temp, on='filePath', how='left')
                toc = time.perf_counter()
                processing_time = toc - tic
                if self.verbose: print(f"{processing_time:.4f} seconds needed")
                
                # Save time needed
                df_logs.at[idx, 'seconds_needed'] = processing_time

                # Save interim df_temp and df_logs
                self.save_results(df_temp)
                self.save_logs(df_logs)

                # Run GC and empty cuda cache
                if self.cuda_availability:
                    torch.cuda.empty_cache()
                gc.collect()
                
                processed_count += 1
                
                if progress_callback:
                    progress_callback({
                        'percentage': (processed_count+1) / len(active_functions),
                        'current_function': func_name,
                        'status': 'completed',
                        'message': f'Completed {func_name} in {processing_time:.2f} seconds'
                    })
            else:
                if progress_callback:
                    progress_callback({
                        'percentage': (processed_count+1) / len(active_functions),
                        'current_function': func_name,
                        'status': 'skipped',
                        'message': f'Skipping {func_name} (disabled in configuration)'
                    })

        if progress_callback:
            progress_callback({
                'percentage': 1.0,
                'current_function': 'Finalizing',
                'status': 'finalizing',
                'message': 'Saving final results and generating output files'
            })

        print(f"### Finished batch of n={len(df_images)} images ###")

        # Output final results
        self._csv_to_xlsx(
            csv_path=os.path.join(self.output_dir, 'results.csv'), 
            xlsx_path=os.path.join(self.output_dir, 'results.xlsx'))
        self._csv_to_xlsx(
            csv_path=os.path.join(self.output_dir, 'logs.csv'), 
            xlsx_path=os.path.join(self.output_dir, 'logs.xlsx'))

        if self.incl_summary_stats:
            self._save_summary_stats(
                df = pd.read_csv(os.path.join(self.output_dir, 'results.csv')),
                csv_path = os.path.join(self.output_dir, 'summary_statistics.csv'),
                xlsx_path = os.path.join(self.output_dir, 'summary_statistics.xlsx'))
        print(f"### Final Excel versions saved to: {self.output_dir} ###")
        
        if progress_callback:
            progress_callback({
                'percentage': 1.0,
                'current_function': 'Complete',
                'status': 'completed',
                'message': f'Successfully processed {len(df_images)} images'
            })
        
        return df_out, df_logs

    def save_logs(self, df_logger, output_dir=None):
        """
        Save the logs to CSV file. If file already exists,
        merge the new logs with existing data.

        :param df_logger: DataFrame containing logs to save
        :param output_dir: The directory to save the logs to. If not provided,
                          the results will be saved in the default output directory.
        """
        if output_dir is None:
            output_dir = self.output_dir

        csv_path = os.path.join(output_dir, 'logs.csv')

        # Create a copy of the dataframe to avoid modifying the original
        df_to_save = df_logger.copy()

        # Check if the functions column contains actual functions or strings
        if df_to_save['functions'].dtype == 'O' and callable(df_to_save['functions'].iloc[0]):
            # Convert function objects to names only when saving
            df_to_save['functions'] = df_to_save['functions'].apply(lambda x: x.__name__ if x else None)

        # Check if file already exists
        if os.path.exists(csv_path):
            # Load existing data
            df_saved = pd.read_csv(csv_path)
            
            # Update existing rows and append new ones
            for idx, row in df_to_save.iterrows():
                func_name = row['functions']
                mask = df_saved['functions'] == func_name
                
                if mask.any():
                    # Update existing row
                    for col in df_to_save.columns:
                        if col != 'functions' and pd.notna(row[col]):
                            df_saved.loc[mask, col] = row[col]
                else:
                    # Append new row
                    df_saved = pd.concat([df_saved, pd.DataFrame([row])], ignore_index=True)

            # Save updated data
            df_saved.to_csv(csv_path, index=False, encoding='utf-8')
        else:
            # Initial save
            df_to_save.to_csv(csv_path, index=False, encoding='utf-8')

    def save_results(self, df_results, output_dir=None):
        """
        Save the processed results to CSV file. If file already exists,
        merge the new results with existing data.

        :param df_results: DataFrame containing results to save
        :param output_dir: The directory to save the results to. If not provided,
                          the results will be saved in the default output directory.
        """
        if output_dir is None:
            output_dir = self.output_dir

        csv_path = os.path.join(output_dir, 'results.csv')

        # Ensure filePath column is string type
        df_results['filePath'] = df_results['filePath'].astype(str)

        # Check if file already exists
        if os.path.exists(csv_path):
            # Load existing data
            df_saved = pd.read_csv(csv_path)
            
            # Ensure filePath column is string type in existing data
            df_saved['filePath'] = df_saved['filePath'].astype(str)
            
            # Merge new results with existing data
            df_merged = df_saved.merge(df_results, on='filePath', how='left')
            
            # Save updated CSV
            df_merged.to_csv(csv_path, index=False, encoding='utf-8')
        else:
            # Initial save
            df_results.to_csv(csv_path, index=False, encoding='utf-8')

    def _csv_to_xlsx(self, csv_path, xlsx_path):
        """
        Helper method to convert a CSV file to an Excel file.
        """
        # Handle results Excel file
        if not os.path.exists(csv_path):
            print(f"### Couldn't generate XLSX from CSV because we couldn't find the CSV file: {csv_path} ###")
        else:
            
            # Open file
            df = pd.read_csv(csv_path)

            # Handle logs and results
            if csv_path.endswith('logs.csv'):
                with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Raw Data', index=False)

            elif csv_path.endswith('results.csv'):
                with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                    # Check Excel column limit
                    if df.shape[1] > 16384:
                        message_df = pd.DataFrame({
                            'Message': ['The results contain more than 16,384 columns, which exceeds Excel\'s limit.',
                                    'Please refer to the CSV file for the complete results.']
                        })
                        message_df.to_excel(writer, sheet_name='Raw Data', index=False)
                    else:
                        df.to_excel(writer, sheet_name='Raw Data', index=False)
            else:
                print(f"### CSV file NOT found: {csv_path} ###")

    def _save_summary_stats(self, df, csv_path, xlsx_path):
        """
        Helper method to calculate and save summary statistics.
        
        :param df: DataFrame to calculate statistics for
        :param writer: ExcelWriter object to save to
        """
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            summary_stats = df[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
        else:
            summary_stats = pd.DataFrame()

        # Binary columns
        binary_cols = df.select_dtypes(include=['bool']).columns
        if len(binary_cols) > 0:
            binary_stats = df[binary_cols].agg(['sum', 'count'])
            binary_stats.loc['share'] = binary_stats.loc['sum'] / binary_stats.loc['count']
            
            # Combine numeric and binary statistics if both exist
            if not summary_stats.empty:
                combined_stats = pd.concat([summary_stats.transpose(), binary_stats.transpose()], axis=0)
            else:
                combined_stats = binary_stats.transpose()
        else:
            # If no binary columns, just use the numeric stats
            combined_stats = summary_stats.transpose() if not summary_stats.empty else pd.DataFrame()
        
        # Only save statistics if we have any
        if not combined_stats.empty:
            # Reset index to make feature names a column
            combined_stats = combined_stats.reset_index()
            combined_stats.columns = ['feature_name'] + list(combined_stats.columns[1:])
            
            # Save with feature names as first column
            combined_stats.to_csv(csv_path, index=False, encoding='utf-8')
            combined_stats.to_excel(xlsx_path, index=False)

    def create_argmin_argmax_pdf(self, results_csv_path = None, output_pdf_path = None, exclude_object_detection_features = False):
        """
        Generate a PDF from a results file that displays the argmin and argmax image per feature.

        :param results_csv_path: The path to the results CSV file.
        :param output_pdf_path: The path to save the PDF file.
        :param exclude_object_detection_features: If True, exclude features that detect objects (i.e., coco_*, imagenet_*, contains_*)
        """        
        
        if results_csv_path is None:
            print(f"### Results file not specified. Using data from current run: {results_csv_path} ###")
            results_csv_path = os.path.join(self.output_dir, 'results.csv')
        if output_pdf_path is None:
            output_pdf_path = os.path.join(self.output_dir, 'argmin_argmax_images_per_feature.pdf')

        # Load results file
        df = pd.read_csv(results_csv_path)

        # Initialize PDF
        pdf = FPDF(orientation='L', unit='mm', format='letter')
        # Get page dimensions (letter size in landscape)
        page_width = 279.4  # mm
        page_height = 215.9  # mm
        margin = 10  # mm
        gap = 5  # mm gap between left and right halves
        half_width = (page_width - 2 * margin - gap) / 2  # Adjusted to account for gap

        # Helper function to sanitize text for PDF encoding
        def sanitize_text(text):
            """Convert text to ASCII-safe string for PDF encoding"""
            if text is None:
                return ""
            # Convert to string and encode/decode to remove non-ASCII characters
            text_str = str(text)
            try:
                # Try to encode as ASCII, replace non-ASCII with '?'
                return text_str.encode('ascii', errors='replace').decode('ascii')
            except:
                # Fallback: replace problematic characters
                return ''.join(c if ord(c) < 128 else '?' for c in text_str)

        # Iterate over each feature
        features = df.columns[1:]  # Assuming first column is filePath
        # Exlude features that are not of interest
        features = [feature for feature in features if feature not in ['fileName', 'fileType', 'descrBlip', 'descrLLM', 'error_llm', 'ocrText', 'ocrLanguage']]
        
        # Exclude coco_ and imagenet_ features if requested
        if exclude_object_detection_features:
            features = [feature for feature in features if not (feature.startswith('coco_') or feature.startswith('imagenet_') or feature.startswith('contains_'))]
        for feature in features:
            pdf.add_page()

            # Set title
            pdf.set_font("Arial", size=12)
            # Sanitize feature name for PDF encoding
            sanitized_feature = sanitize_text(feature)
            pdf.cell(0, 10, txt=f"Feature: {sanitized_feature}", ln=True, align='C')

            # Find min and max images for the feature
            min_image = df.loc[df[feature].idxmin()]['filePath']
            max_image = df.loc[df[feature].idxmax()]['filePath']

            def scale_image(img_path):
                """Helper function to calculate scaled dimensions"""
                if not os.path.exists(img_path):
                    return None, None
                
                from PIL import Image
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    img_aspect = img_w / img_h
                    
                    # Calculate dimensions to fit half page width and full height
                    width_based = half_width
                    height_based = page_height - 40  # Account for margins and text
                    
                    # Calculate both possible dimensions
                    if width_based / img_aspect <= height_based:
                        # Width is the limiting factor
                        return width_based, width_based / img_aspect
                    else:
                        # Height is the limiting factor
                        return height_based * img_aspect, height_based

            # Add image labels on same line
            pdf.set_font("Arial", size=10)
            min_value = df.loc[df[feature].idxmin()][feature]
            max_value = df.loc[df[feature].idxmax()][feature]
            
            # Calculate widths for left and right text cells
            # Check if values are numeric before formatting with .4f
            if isinstance(min_value, (int, float)):
                left_text = f"Min Image: {sanitize_text(os.path.basename(min_image))} (Value: {min_value:.4f})"
            else:
                left_text = f"Min Image: {sanitize_text(os.path.basename(min_image))} (Value: {sanitize_text(min_value)})"
                
            if isinstance(max_value, (int, float)):
                right_text = f"Max Image: {sanitize_text(os.path.basename(max_image))} (Value: {max_value:.4f})"
            else:
                right_text = f"Max Image: {sanitize_text(os.path.basename(max_image))} (Value: {sanitize_text(max_value)})"
            
            # Print both texts on same line, both left-aligned
            pdf.cell(half_width + margin, 10, txt=left_text, ln=0, align='L')
            pdf.cell(half_width + margin, 10, txt=right_text, ln=1, align='L')

            # Store the Y position after text for both images to ensure alignment
            image_start_y = pdf.get_y()

            # Add images
            if os.path.exists(min_image):
                # TODO: Implement webp support
                w, h = scale_image(min_image)
                if w and h:
                    # Center in left half
                    x = margin + (half_width - w) / 2
                    y = image_start_y + (page_height - image_start_y - h) / 2
                    pdf.image(min_image, x=x, y=y, w=w, h=h)

            if os.path.exists(max_image):
                # TODO: Implement webp support
                w, h = scale_image(max_image)
                if w and h:
                    # Center in right half (with gap)
                    x = margin + half_width + gap + (half_width - w) / 2
                    y = image_start_y + (page_height - image_start_y - h) / 2
                    pdf.image(max_image, x=x, y=y, w=w, h=h)

            # Add footer text
            pdf.set_y(2)  # Move to 20mm from bottom
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt=sanitize_text("Images analyzed"), ln=True, align='C')

        # Save the PDF
        pdf.output(output_pdf_path)
        print(f"### PDF saved to: {output_pdf_path} ###")