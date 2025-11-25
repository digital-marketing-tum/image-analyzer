# Image Analyzer

Image Analyzer is an end-to-end image analysis tool designed around the foundational concepts of comprehensiveness, open source, and transparency. It supports the holistic extraction of a broad and diverse range of image features, from low-level pixel properties to high-level semantic content, allowing its users to capture the full richness of visual data.

This tool is based on our [Image Analyzer](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5776702) paper. Please refer to the paper or the code if you have questions on how individual features are derived.

The easisest way to use Image Analyzer is via Google Colab. 
To do so, simply click the Google Colab button below and follow the instructions in our notebook.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digital-marketing-tum/image-analyzer/blob/main/src/notebooks/pipeline_colab.ipynb)

Alternatively, you can install and use Image Analyzer on your local machine which gives you greater control and much more flexibility. To do so, please follow the instructions in this document.

## Local installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (optional, for faster processing)
- Tesseract OCR (for text recognition features)

**⚠️ Image Analyzer downloads model weights:**
Several features in this pipeline require downloading large pre-trained model weights. These downloads happen automatically when you first use these features. Ensure you have enough disk space. 

Follow these steps to install Image Analyzer and all its dependencies on your local machine:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/digital-marketing-tum/image-analyzer.git
   cd IA
   ```

2. **Create and activate the virtual environment**:

   ```bash
   python -m venv .venv
   
   # Using Linux/Mac:
   source .venv/bin/activate
   
   # Using Windows command line:
   .venv\Scripts\activate

   # Using Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   
   a. PyTorch installation (GPU/CPU Support)

      **Important:** PyTorch is not included in `requirements.txt` to ensure compatibility with all platforms and CUDA versions. <br>
      Before installing the remaining dependencies, please install PyTorch according to your system and CUDA version by following the official instructions at: https://pytorch.org/get-started/locally/

      **Example for CUDA 12.1 (works with CUDA 12.2+ drivers):**
      ```sh
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      ```

      **For CPU-only:**
      ```sh
      pip install torch torchvision torchaudio
      ```

   b. Requirements installation
      
      After installing PyTorch, install the remaining dependencies:
      ```sh
      pip install -r requirements.txt
      ```

## Run Image Analyzer

You have a couple of options to run Image Analyzer locally. You can either use a visual user interface (i.e., web application) or run it in a local notebook (i.e., Jupyter notebook).

### Option 1: Web Application (Currently only supported by Windows)

**⚠️ The web application is currently only supported on Windows devices.**

Before you follow the steps below, ensure the virtual environment is still active. If not, please see above on how to activate it.

1. **Start the web server**

   ```bash
   python app.py
   ```

2. **Access and use the web application**:

   - Open the web application in your browser through http://127.0.0.1:5000
   - Follow the steps outlined in the web application:
      - Step 1: Select and configure the features you want to analyze; ensure you click save at the bottom of the page
      - Step 2: Upload images or folders start Image Analyzer analysis
      - Step 3: Download your results and unzip them locally

### Option 2: Jupyter Notebook

1. **Configure Image Analyzer**

   Open `config/configuration.yaml` and modify it if needed.

   You have to:
   - Adjust the input directory under `config` -> `input_dir` to point towards the directory where your images are saved

   Additionally, you can:
   - Enable/disable specific features by setting `active: true/false`
   - Adjust parameters for each feature
   - Configure processing options (e.g., debug mode)

   ⚠️ For OCR, you need to provide the path to the local tesseract installation so that Image Analyzer can use its engine.

2. **Run Image Analyzer**

   Open `src/notebooks/pipeline_local.ipynb` and run all cells. <br>
   Image Analyzer will print a timestamp to the console. This timestamp serves as identifier for a specific Image Analyzer run.

3. **Access results**

   After completion, you can find your results in the `outputs/Image-Analyzer_run_YYYYMMDD_HHMMSS` directory where YYYYMMDD_HHMMSS refers to the run-specific timestamp.

## External Dependencies

### Tesseract OCR
⚠️ OCR is currently only supported on Windows systems.

**Windows**:
- Download from [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Install to default `C:\Program Files\Tesseract-OCR\` directory
- If you are using a different directory, you can adjust the variable `windows_path_to_tesseract` in the `config/configuration.yaml` file

### OpenAI API
If you want to use the `describe_llm_api` feature extraction you need to specify an OpenAI key in the `config/configuration.yaml` file.
⚠️ Don't commit your `config/configuration.yaml` file containing your API key!

## Contributing

Contributions of all experience levels are welcome! There are many ways to contribute, and we appreciate your help.<br>
To contribute, please:
1. Fork the repository
2. Create a feature branch
3. Make your changes and document them
4. Update the environment file if needed:
   ```bash
   pip freeze > requirements.txt
   ```
5. Submit a pull request

If you would like to implement a new feature, please consult our template function for contributers `new_feature_template()` which you can find in `src/image_analyzer/utils/feature_extraction.py`

## License and citation

This repository is published under a GNU AGPLv3 license. For details, please refer to the `LICENSE.md` file.

If you use Image Analyzer, we kindly ask you to cite our corresponding academic paper:

```bibtex
@article{Image_Analyzer,
  author={Exner, Yannick and Konrad, Maximilian and Konrad, Maximilian and Hartmann, Jochen},  
  title={{Image Analyzer: A Framework and Pipeline to Analyze Image Metrics}},
  year={2025},
  journal = {{SSRN Electronic Journal}},
  doi={10.2139/ssrn.5776702}
}
```

## 🙏 Thank You

Thanks to all our amazing contributors 💪  
[@karolina-tum](https://github.com/karolina-tum) · [@leokinzinger](https://github.com/leokinzinger) · [@maximilian-konrad](https://github.com/maximilian-konrad) · [@NicolasLupke](https://github.com/NicolasLupke) · [@samueldomdey](https://github.com/samueldomdey) · [@satyam-kr03](https://github.com/satyam-kr03) · [@tqm-111](https://github.com/tqm-111) · [@yexner](https://github.com/yexner) · [@zhangsh1416](https://github.com/zhangsh1416)
