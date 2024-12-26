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

# Add at the start of your script
import dotenv
dotenv.load_dotenv()

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
        self.last_process_time = time.time()
        self.process_every = 0.1  # Process every 100ms
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        # Save the current frame for capture regardless of processing
        self.frame = img.copy()
        
        # Only process every 100ms to reduce computational load
        if current_time - self.last_process_time > self.process_every:
            # Run detection
            results = self.model.predict(img, conf=self.confidence)[0]
            
            # Update detections list
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
            
            # Draw detections
            annotated_frame = results.plot()
            self.last_process_time = current_time
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        
        # If not processing this frame, return the original
        return av.VideoFrame.from_ndarray(img, format="bgr24")

@st.cache_data
def get_ice_servers():
    ice_servers = [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
                "stun:stun2.l.google.com:19302"
            ]
        }
    ]
    
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        
        if account_sid and auth_token:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
            
    except (KeyError, Exception) as e:
        logger.warning(
            "Using public STUN servers. For better connectivity, configure Twilio credentials."
        )
    
    return ice_servers

def infer_uploaded_webcam(conf, model):
    """
    Enhanced webcam detection implementation using streamlit-webrtc
    """
    st.write("### Real-time Underwater Object Detection")
    
    # Initialize session state for captured frames
    if 'captured_frames' not in st.session_state:
        st.session_state.captured_frames = []

    # Create columns for controls
    col1, col2 = st.columns(2)
    
    # Clear captures button
    if col2.button("Clear All Captures"):
        st.session_state.captured_frames = []
        st.experimental_rerun()

    # WebRTC Configuration
    rtc_config = {"iceServers": get_ice_servers()}
    
    # Initialize the WebRTC streamer with improved configuration
    ctx = webrtc_streamer(
        key="underwater-object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_transformer_factory=lambda: VideoTransformer(model, conf),
        async_transform=True,
        media_stream_constraints={
            "video": {"frameRate": {"ideal": 30}},
            "audio": False
        }
    )

    # Capture button
    if ctx.video_transformer and col1.button("Capture Frame"):
        try:
            if hasattr(ctx.video_transformer, 'frame'):
                # Create timestamp and ensure directory exists
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                capture_dir = Path("captures")
                capture_dir.mkdir(exist_ok=True)
                
                # Get current frame
                frame = ctx.video_transformer.frame
                if frame is not None:
                    # Save original frame
                    original_path = capture_dir / f"frame_original_{timestamp}.jpg"
                    cv2.imwrite(str(original_path), frame)
                    
                    # Process frame with detections
                    results = ctx.video_transformer.model.predict(frame, conf=ctx.video_transformer.confidence)[0]
                    processed_frame = results.plot()
                    
                    # Save processed frame
                    processed_path = capture_dir / f"frame_processed_{timestamp}.jpg"
                    cv2.imwrite(str(processed_path), processed_frame)
                    
                    # Store in session state
                    st.session_state.captured_frames.append({
                        'timestamp': timestamp,
                        'processed': str(processed_path),
                        'original': str(original_path),
                        'detections': [
                            {
                                'class': det.class_name,
                                'confidence': det.confidence
                            }
                            for det in ctx.video_transformer.detections
                        ]
                    })
                    
                    st.success("Frame captured successfully!")
                else:
                    st.error("No frame available to capture")
        except Exception as e:
            st.error(f"Error capturing frame: {str(e)}")
    
    # Display captured frames
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
                
                # Display detections
                st.write("#### Detections:")
                st.write(f"Total objects detected: {len(frame_info['detections'])}")
                for detection in frame_info['detections']:
                    st.write(f"🎯 {detection['class']}: {detection['confidence']:.2%}")