# utils.py
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import torch
import time
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
from typing import List, NamedTuple
from twilio.rest import Client
import logging
import dotenv
from pathlib import Path
from collections import deque

dotenv.load_dotenv()


# Configure logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    Sets device to GPU if available, otherwise CPU.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    st.info(f"Using device: {device}")
    return model

def _display_detected_frames(conf, model, st_frame, image):
    """
    Process and display detected frames with detection information.
    Returns detection statistics for tracking.
    """
    start_time = time.time()
    
    # Resize the image to maintain aspect ratio
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    
    # Predict with the model
    res = model.predict(image, conf=conf)[0]
    res_plotted = res.plot()
    
    # Process detection results
    boxes = res.boxes
    total_objects = len(boxes)
    detection_summary = []
    
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        detection_summary.append(f"{cls_name}: {conf:.2%}")
    
    processing_time = time.time() - start_time
    
    # Display the processed frame
    st_frame.image(res_plotted, caption='Detected Frame', channels="BGR", use_column_width=True)
    
    return total_objects, detection_summary, processing_time

def infer_uploaded_image(conf, model):
    """
    Process uploaded images and display detection results.
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            st.image(source_img, caption="Uploaded Image", use_column_width=True)

    if source_img and st.button("Execute Detection"):
        with st.spinner("Processing image..."):
            start_time = time.time()
            
            # Run detection
            res = model.predict(uploaded_image, conf=conf)[0]
            res_plotted = res.plot()[:, :, ::-1]
            
            processing_time = time.time() - start_time
            
            with col2:
                st.image(res_plotted, caption="Detected Image", use_column_width=True)
                
                # Display detection statistics
                st.write("### Detection Results")
                total_objects = len(res.boxes)
                st.write(f"📊 Total objects detected: {total_objects}")
                st.write(f"⏱️ Processing time: {processing_time:.2f} seconds")
                
                # Show detailed detection results
                with st.expander("Detailed Detection Results"):
                    for box in res.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        st.write(f"🎯 {cls_name}: {conf:.2%}")

def process_video_frame(conf, model, frame):
    """
    Process a single video frame and return detection results.
    """
    # Ensure frame is in correct format
    if isinstance(frame, Image.Image):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    
    # Run detection
    results = model.predict(frame, conf=conf)[0]
    return results.plot()

def process_video_frame(conf, model, frame):
    """
    Process a single video frame and return detection results.
    """
    if isinstance(frame, Image.Image):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    
    results = model.predict(frame, conf=conf)[0]
    return results.plot(), results

def infer_uploaded_video(conf, model):
    """
    Process uploaded videos with state management and fixed file handling.
    """
    source_video = st.sidebar.file_uploader(label="Choose a video...", type=['mp4', 'avi', 'mov'])
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
        
    if source_video:
        st.write("### Original Video")
        st.video(source_video)
        
        if st.button("Process Video"):
            try:
                with st.spinner("Processing video... This may take a few minutes."):
                    # Create temporary directory
                    temp_dir = tempfile.mkdtemp()
                    input_path = os.path.join(temp_dir, 'input.mp4')
                    output_path = os.path.join(temp_dir, 'output.mp4')
                    
                    # Save uploaded video to temporary file
                    with open(input_path, 'wb') as f:
                        f.write(source_video.getbuffer())
                    
                    # Video capture setup
                    vid_cap = cv2.VideoCapture(input_path)
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Video writer setup
                    temp_output = os.path.join(temp_dir, 'temp_output.mp4')
                    video_writer = cv2.VideoWriter(
                        temp_output,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (width, height)
                    )
                    
                    # Process frames
                    progress_bar = st.progress(0)
                    frame_count = 0
                    
                    while vid_cap.isOpened():
                        success, frame = vid_cap.read()
                        if not success:
                            break
                            
                        processed_frame, _ = process_video_frame(conf, model, frame)
                        video_writer.write(processed_frame)
                        
                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames)
                    
                    # Clean up
                    vid_cap.release()
                    video_writer.release()
                    
                    # Convert video to MP4 with H.264 codec using FFmpeg
                    os.system(f'ffmpeg -i {temp_output} -vcodec libx264 {output_path}')
                    
                    if os.path.exists(output_path):
                        st.session_state.output_path = output_path
                        st.session_state.processing_complete = True
                        
                        # Display processed video
                        st.success("Video processing completed!")
                        st.video(output_path)
                        
                        # Offer download
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download Processed Video",
                                data=f.read(),
                                file_name="detected_video.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("Error saving the processed video.")
                        
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")


class Detection(NamedTuple):
    class_name: str
    confidence: float
    bbox: tuple

class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, confidence):
        self.model = model
        self.confidence = confidence
        self.detections: List[Detection] = []
        self.frame = None
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img.copy()
        results = self.model.predict(img, conf=self.confidence)[0]
        
        self.detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            
            self.detections.append(Detection(
                class_name=self.model.names[class_id],
                confidence=conf,
                bbox=bbox
            ))
        
        annotated_frame = results.plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def infer_uploaded_webcam_cloud(conf, model):
    """
    Cloud version of webcam detection - shows a friendly message
    """
    st.info("🎥 Webcam Detection Feature")
    st.warning("""
        This feature is currently optimized for local deployment and may not work correctly in the cloud environment.
        
        To use the webcam detection feature:
        1. Clone the repository
        2. Install the required dependencies
        3. Run the application locally using `streamlit run app.py`
        
        We're working on making this feature available in the cloud. Thank you for your understanding!
    """)
    
    # Optional: Add a link to the repository or documentation
    st.markdown("[📚 Learn more about local deployment](your-repo-link-here)")

def get_available_cameras():
    """Test available camera indices and return valid ones."""
    available_cameras = []
    for i in range(10):  # Test first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def infer_uploaded_webcam_local(conf, model):
    st.write("### Real-time Object Detection")
    
    
    if 'captured_frames' not in st.session_state:
    st.session_state.captured_frames = []

    # Get available cameras
    available_cameras = get_available_cameras()
    if not available_cameras:
        st.error("No cameras detected")
        return
    
    camera_index = st.selectbox(
        "Select Camera",
        available_cameras,
        format_func=lambda x: f"Camera {x}"
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access webcam")
        return

    col1, col2 = st.columns(2)
    
    if col2.button("Clear Captures"):
        st.session_state.captured_frames = []
        st.experimental_rerun()

    FRAME_WINDOW = st.image([])
    capture_button = col1.button("Capture Frame")
    stop_button = st.button("Stop")

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf)[0]
        processed_frame = results.plot()
        
        FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

        if capture_button:
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                os.makedirs("captures", exist_ok=True)
                
                original_path = os.path.join("captures", f"frame_original_{timestamp}.jpg")
                processed_path = os.path.join("captures", f"frame_processed_{timestamp}.jpg")
                
                cv2.imwrite(original_path, frame)
                cv2.imwrite(processed_path, processed_frame)
                
                st.session_state.captured_frames.append({
                    'timestamp': timestamp,
                    'processed': processed_path,
                    'original': original_path,
                    'detections': [
                        {
                            'class': model.names[int(box.cls[0])],
                            'confidence': float(box.conf[0])
                        }
                        for box in results.boxes
                    ]
                })
                st.success("Frame captured!")
                break
            except Exception as e:
                st.error(f"Error capturing frame: {str(e)}")
                break

    cap.release()

    if st.session_state.captured_frames:
        st.write("### Captured Frames")
        for frame_info in reversed(st.session_state.captured_frames):
            with st.expander(f"Capture at {frame_info['timestamp']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Original Frame")
                    if os.path.exists(frame_info['original']):
                        st.image(frame_info['original'])
                        with open(frame_info['original'], 'rb') as f:
                            st.download_button(
                                "Download Original",
                                f.read(),
                                f"original_{frame_info['timestamp']}.jpg",
                                "image/jpeg"
                            )
                
                with col2:
                    st.write("Detected Frame")
                    if os.path.exists(frame_info['processed']):
                        st.image(frame_info['processed'])
                        with open(frame_info['processed'], 'rb') as f:
                            st.download_button(
                                "Download Processed",
                                f.read(),
                                f"processed_{frame_info['timestamp']}.jpg",
                                "image/jpeg"
                            )
                
                st.write("#### Detections:")
                st.write(f"Total objects detected: {len(frame_info['detections'])}")
                for detection in frame_info['detections']:
                    st.write(f"🎯 {detection['class']}: {detection['confidence']:.2%}")
                    