"""
Demo Application for License Plate Recognition System
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from models.plate_recognition_system import PlateRecognitionSystem

def process_single_image(image_path, output_dir="src/output"):
    """
    Process a single license plate image
    
    Args:
        image_path: Path to the license plate image
        output_dir: Directory to save the output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the system
    system = PlateRecognitionSystem()
    
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Process the plate
    results = system.process_plate(img)
    plate_number, plate_type, confidence = results
    
    print("\nLicense Plate Recognition Results:")
    print("=" * 35)
    print(f"Plate Number: {plate_number}")
    print(f"Plate Type: {plate_type}")
    print(f"Confidence: {confidence:.1f}%")
    
    # Visualize results
    output_path = os.path.join(output_dir, f"demo_result_{os.path.basename(image_path)}")
    vis_img, _ = system.visualize_results(img, output_path)
    
    print(f"\nVisualization saved to: {output_path}")
    
    # Display the image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Plate: {plate_number} | Type: {plate_type} ({confidence:.1f}%)")
    plt.tight_layout()
    
    # Save the matplotlib figure
    plt_output_path = os.path.join(output_dir, f"demo_plot_{os.path.basename(image_path)}")
    plt.savefig(plt_output_path)
    print(f"Plot saved to: {plt_output_path}")

def process_batch(input_dir, output_dir="src/output"):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing license plate images
        output_dir: Directory to save the output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the system
    system = PlateRecognitionSystem()
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results_list = []
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_dir, image_file)
        
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            continue
        
        # Process the plate
        results = system.process_plate(img)
        results['filename'] = image_file
        results_list.append(results)
        
        # Visualize results
        output_path = os.path.join(output_dir, f"batch_result_{image_file}")
        vis_img, _ = system.visualize_results(img, output_path)
        
        print(f"Processed {i+1}/{len(image_files)}: {image_file}")
    
    # Create a summary visualization
    create_summary_visualization(results_list, input_dir, output_dir)

def create_summary_visualization(results_list, input_dir, output_dir):
    """
    Create a summary visualization of all processed images
    
    Args:
        results_list: List of results dictionaries
        input_dir: Directory containing input images
        output_dir: Directory to save the output
    """
    # Determine grid size
    n_images = len(results_list)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Create figure
    plt.figure(figsize=(15, 15))
    
    # Plot each image
    for i, results in enumerate(results_list):
        if i >= grid_size * grid_size:
            break
            
        # Read image
        image_path = os.path.join(input_dir, results['filename'])
        img = cv2.imread(image_path)
        
        if img is None:
            continue
            
        # Add to plot
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{results['plate_number']}\n{results['plate_type']}")
        plt.axis('off')
    
    # Save the summary visualization
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "batch_summary.jpg")
    plt.savefig(summary_path)
    print(f"\nSummary visualization saved to: {summary_path}")

def create_demo_ui():
    """
    Create a simple command-line UI for the demo application
    """
    print("\n" + "=" * 50)
    print("License Plate Recognition System - Demo Application")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Process a single image")
        print("2. Process all images in a directory")
        print("3. Process sample images from the dataset")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            image_path = input("Enter the path to the image: ")
            if os.path.exists(image_path):
                process_single_image(image_path)
            else:
                print(f"Error: File not found at {image_path}")
        
        elif choice == '2':
            input_dir = input("Enter the directory containing images: ")
            if os.path.isdir(input_dir):
                process_batch(input_dir)
            else:
                print(f"Error: Directory not found at {input_dir}")
        
        elif choice == '3':
            # Process sample images from each category
            categories = ['green', 'blue', 'red']
            for category in categories:
                category_dir = f"src/data/images/{category}"
                if os.path.exists(category_dir):
                    # Get the first image
                    image_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if image_files:
                        image_path = os.path.join(category_dir, image_files[0])
                        print(f"\nProcessing sample {category} plate:")
                        process_single_image(image_path)
            
            # Process a test image
            test_dir = "src/data/images/test"
            if os.path.exists(test_dir):
                image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    image_path = os.path.join(test_dir, image_files[0])
                    print(f"\nProcessing sample test plate:")
                    process_single_image(image_path)
        
        elif choice == '4':
            print("\nExiting the demo application. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="License Plate Recognition System Demo")
    parser.add_argument("--image", help="Path to a single license plate image")
    parser.add_argument("--dir", help="Directory containing license plate images")
    parser.add_argument("--samples", action="store_true", help="Process sample images from the dataset")
    parser.add_argument("--output", default="src/output", help="Directory to save output")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.image:
        # Process a single image
        process_single_image(args.image, args.output)
    
    elif args.dir:
        # Process all images in a directory
        process_batch(args.dir, args.output)
    
    elif args.samples:
        # Process sample images from each category
        categories = ['green', 'blue', 'red']
        for category in categories:
            category_dir = f"src/data/images/{category}"
            if os.path.exists(category_dir):
                # Get the first image
                image_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    image_path = os.path.join(category_dir, image_files[0])
                    print(f"\nProcessing sample {category} plate:")
                    process_single_image(image_path, args.output)
        
        # Process a test image
        test_dir = "src/data/images/test"
        if os.path.exists(test_dir):
            image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                image_path = os.path.join(test_dir, image_files[0])
                print(f"\nProcessing sample test plate:")
                process_single_image(image_path, args.output)
    
    else:
        # No arguments provided, launch interactive UI
        create_demo_ui()

if __name__ == "__main__":
    main()