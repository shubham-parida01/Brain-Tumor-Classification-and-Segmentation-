# Brain Tumor Detection Model Test Suite

This directory contains test implementations for the brain tumor detection model components.

## Components Tested

1. **DWT Preprocessing**
   - Discrete Wavelet Transform for image denoising
   - Configurable wavelet type and decomposition levels
   - Visualization of original vs. preprocessed images

2. **CSA Feature Selection**
   - Crow Search Algorithm for feature selection
   - Configurable population size and iterations
   - Visualization of selected features and fitness history

3. **Attention Mechanisms**
   - Multi-scale channel and spatial attention
   - Feature fusion at multiple scales
   - Visualization of attention maps

4. **Visualization Tools**
   - Model architecture visualization
   - Training metrics plotting
   - Confusion matrix and ROC curves
   - Segmentation results visualization

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

To run all tests:
```bash
python test_implementations.py
```

The test script will:
1. Test each component individually
2. Generate visualizations in the `test_drive` directory
3. Save TensorBoard logs in `test_drive/logs`

## Expected Output

After running the tests, you should see:
- Success messages for each component test
- Visualization files in the `test_drive` directory:
  - `dwt_test.png`: DWT preprocessing results
  - `csa_test.png`: CSA feature selection results
  - `attention_test.png`: Attention mechanism results
  - `model_architecture.png`: Model architecture visualization
  - `attention_maps.png`: Attention maps visualization
  - `training_metrics.png`: Training metrics plot
  - `confusion_matrix.png`: Confusion matrix
  - `roc_curves.png`: ROC curves
  - `segmentation_results.png`: Segmentation results

## Notes

- The tests use synthetic data for demonstration purposes
- For real-world testing, replace the synthetic data with actual brain MRI images
- Adjust parameters in the test script to match your specific requirements 