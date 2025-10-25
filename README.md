# AI-Powered Automated Car Plate Number Recognition System

## Project Overview

This project implements an AI-powered license plate recognition system that can:
1. Extract alphanumeric text from car license plates using OCR
2. Identify the type of plate based on its background color:
   - Green: Government vehicles
   - Blue: Private vehicles
   - Red: Commercial vehicles

## Research Gap

Current license plate recognition systems typically focus only on extracting the alphanumeric characters from license plates. This project extends that functionality by adding color-based classification to determine the vehicle category, which provides additional contextual information that can be valuable for:

- Law enforcement agencies to quickly identify vehicle types
- Automated tolling systems to apply different rates based on vehicle categories
- Parking management systems to allocate spaces based on vehicle types
- Traffic monitoring and analysis

## System Architecture

The system consists of two main components:

1. **OCR Module**: Extracts the alphanumeric plate number using Tesseract OCR with preprocessing techniques.
2. **Color Classification Module**: Identifies the plate category by analyzing the dominant background color using HSV color space analysis.

### Technical Approach

The system follows a two-stage pipeline:
1. **Color Classification**: Convert the image to HSV color space and analyze the dominant color to determine the plate type.
2. **OCR Processing**: Preprocess the image and use Tesseract OCR to extract the alphanumeric characters.

## Directory Structure

```
.
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── src/
│   ├── data/                 # Dataset directory
│   │   ├── images/           # License plate images
│   │   │   ├── green/        # Government plates
│   │   │   ├── blue/         # Private plates
│   │   │   ├── red/          # Commercial plates
│   │   │   └── test/         # Test images
│   │   └── ground_truth.json # Ground truth data for evaluation
│   ├── models/               # Model implementations
│   │   ├── color_classifier.py      # Color classification module
│   │   ├── ocr_module.py            # OCR module
│   │   └── plate_recognition_system.py # Integrated system
│   ├── output/               # Output directory for results
│   └── utils/                # Utility scripts
│       ├── dataset_preparation.py    # Dataset preparation utilities
│       ├── evaluation.py             # Evaluation framework
│       ├── improved_evaluation.py    # Improved evaluation framework
│       ├── create_ground_truth.py    # Ground truth creation script
│       └── visualize_samples.py      # Visualization utilities
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/license-plate-recognition.git
cd license-plate-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
```bash
# For Ubuntu/Debian
sudo apt-get install tesseract-ocr

# For macOS
brew install tesseract

# For Windows
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Preparing the Dataset

To generate synthetic license plate images for testing:

```bash
python src/utils/dataset_preparation.py
```

### Running the System

To process a single license plate image:

```python
from src.models.plate_recognition_system import PlateRecognitionSystem

# Initialize the system
system = PlateRecognitionSystem()

# Process a license plate image
image = cv2.imread('path/to/license_plate.jpg')
results = system.process_plate(image)

# Print results
print(f"Plate Number: {results['plate_number']}")
print(f"Plate Type: {results['plate_type']} ({results['confidence']:.1f}%)")

# Visualize results
vis_image, _ = system.visualize_results(image, 'output.jpg')
```

### Evaluating the System

To evaluate the system on the entire dataset:

```bash
python src/utils/improved_evaluation.py
```

## Performance

The system achieves the following performance metrics:

1. **Color Classification**:
   - Accuracy: 100%
   - Precision, Recall, and F1-Score: 100% for all classes

2. **OCR Performance**:
   - Character-level Accuracy: ~5.75%
   - The OCR performance is currently limited and can be improved with better preprocessing and fine-tuning.

## Future Improvements

1. **OCR Enhancement**:
   - Implement more advanced preprocessing techniques
   - Fine-tune Tesseract OCR parameters for license plates
   - Consider using deep learning-based OCR models

2. **Color Classification Robustness**:
   - Add support for handling varying lighting conditions
   - Implement histogram equalization for better color consistency
   - Consider using CNN-based classification for more robust results

3. **System Integration**:
   - Integrate with a license plate detection system to process full vehicle images
   - Develop a web or mobile interface for real-time processing
   - Add support for video processing for traffic monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- Tesseract OCR for text extraction
- NumPy and scikit-learn for data processing and evaluation