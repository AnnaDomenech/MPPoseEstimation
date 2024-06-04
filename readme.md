# Pose Detection with MediaPipe

This program utilizes the MediaPipe model to detect human poses and implement different measurements. From an input image, the script can execute three different scenarios:
- A: Determine shoulder abduction angle.
- B: Determine if the arms are straight or not.
- C: Determine if the pose is facing the camera. 

## Usage

### Command-Line Arguments

- `scenario`: Specifies which scenario (A, B, C) to execute.
- `folder`: Directory to save or load relevant data.
- `show`: Boolean to indicate if the processed video should be displayed.
- `model_path`: Path to the MediaPipe model file.

### Execution

To run the program, use the following command:

```sh
python main.py --scenario <scenario> --folder <folder> --show <show> --model_path <model_path>
```

Replace the placeholders with appropriate values:

- `<scenario>`: A, B, or C. Note that all scenarios implement A.
- `<folder>`: Folder path to the images.
- `<show>`: True or False. If True an image with the results will be displayed. Press any key to end the visualization.
- `<model_path>`: Path to the model file.

### Example

```sh
python pose_detection.py --scenario A --folder ./data/A --show True  --model_path ./models/pose_landmarker_lite.task
```

## Questions
- Scenario B. What can we say about this check? Is it reliable?

Checking if the arm is straight is necessary to ensure an accurate evaluation of the abduction range. However, this metric alone is not sufficiently reliable, as noisy measurements can trigger false warnings.

- What kind of changes would you introduce in the code to make the test work with a video or a camera stream instead from still pictures?

First, the configuration of the model needs to be adjusted, starting with the running mode. Additionally, an approach to handle the different scenarios parallely should be defined. Initially, I would check for a few frames to determine if the person is facing the camera. If this condition is met, I would then evaluate scenarios A and B in parallel. To optimize performance, I would execute these scenarios every 10-12 frames while checking if the initial condition is still met. Computing the mean position of the involved landmarks in batches could help minimize noisy measurements.

- Try the application using images provided by you and check the results.

## References

- MediaPipe: https://mediapipe.dev
- OpenCV: https://opencv.org
- Python: https://www.python.org

