import requests
import streamlit as st
import av
import logging
import os
import tempfile
# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="Ai Classification", page_icon="🤖")
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from streamlit_webrtc import (
	ClientSettings,
	VideoTransformerBase,
	WebRtcMode,
	webrtc_streamer,
)
import supervision as sv
import numpy as np
import glob
import sys
import json
# Define the zone polygon
zone_polygon_m = np.array([[160, 100], 
						 [160, 380], 
						 [481, 380], 
						 [481, 100]], dtype=np.int32)
# # Calculate the center of the polygon
# center = np.mean(zone_polygon_m, axis=0)

# # Compute the vector from the center to each corner
# vectors = zone_polygon_m - center

# # Scale each vector by 20%
# expanded_vectors = vectors * 1.2

# # Added the expanded vectors to the center to get the new corners
# zone_polygon_m = (expanded_vectors + center).astype(np.int32)

try:
	# Get current working directory
	cwd = os.getcwd()

	# Define the path to the detect folder
	detect_path = os.path.join(cwd, "runs")

	# Get a list of all subdirectories in the "runs" directory
	subdirectories_r = [d for d in os.listdir(detect_path) if os.path.isdir(os.path.join(detect_path, d))]

	# Filter out hidden directories (starting with ".")
	subdirectories_r = [d for d in subdirectories_r if not d.startswith(".")]

	# Sort the directories by creation time (most recent first)
	subdirectories_r.sort(key=lambda x: os.path.getmtime(os.path.join(detect_path, x)), reverse=True)

	# Get the path of the most recent directory
	most_recent_folder = subdirectories_r[0]

	# Get the full path of the most recent folder
	most_recent_folder_path = os.path.join(detect_path, most_recent_folder)

	# Get a list of all subdirectories in the detect folder
	subdirectories = [d for d in os.listdir(most_recent_folder_path) if os.path.isdir(os.path.join(most_recent_folder_path, d))]

	# Filter out hidden directories (starting with ".")
	subdirectories = [d for d in subdirectories if not d.startswith(".")]

	# Sort the directories by creation time (most recent first)
	subdirectories.sort(key=lambda x: os.path.getmtime(os.path.join(most_recent_folder_path, x)), reverse=True)

	# Get the path of the most recent directory
	most_recent_folder_full = subdirectories[0]

	# Get the full path of the most recent folder
	most_recent_folder_path_full_1 = os.path.join(most_recent_folder_path, most_recent_folder_full)
	best = "best.pt"
	weights = "weights"
	most_recent_folder_path_weights = os.path.join(most_recent_folder_path_full_1, weights, best)
	# recent_runs_folder_path = get_recent_runs_folder_path()
	st.write(most_recent_folder_path_weights) 
	# Initialize the YOLOv8 model
	# model = YOLO("best (3).pt")
	# @st.cache_resource
	def load_yolo_model():
		return YOLO(most_recent_folder_path_weights)

	# Load the YOLO model (this will be cached)
	model = load_yolo_model()

except FileNotFoundError as e:
	st.warning(f"An error occurred: {e}")
	st.title("⚠️First Train Your Custom Model⚠️")
	sys.exit()
except IndexError:
	st.error("No recent folders found.")
	st.title("⚠️First Train Your Custom Model⚠️")
	sys.exit()

# Initialize the tracker, annotators and zone
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
zone = sv.PolygonZone(polygon=zone_polygon_m, frame_resolution_wh=(642, 642))
mask_annotator = sv.MaskAnnotator()


zone_annotator = sv.PolygonZoneAnnotator(
	zone=zone,
	color=sv.Color.red(),
	thickness=2,
	text_thickness=4,
	text_scale=2
)

def draw_annotations(frame, boxes, masks, names):
	for box, name in zip(boxes, names):
		color = (0, 255, 0)  # Green color for bounding boxes

		# Draw bounding box
		cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

		# Check if masks are available
		if masks is not None:
			mask = masks[frame_number]
			alpha = 0.3  # Transparency of masks

			# Draw mask
			frame[mask > 0] = frame[mask > 0] * (1 - alpha) + np.array(color) * alpha

		# Display class name
		cv2.putText(frame, name, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return frame

# Define the initial confidence threshold


def main():
	st.title("🤖 Ai Classification")
	st.subheader("YOLOv8 & Streamlit WebRTC Integration :)")
	st.sidebar.title("Select an option ⤵️")
	choice = st.sidebar.radio("", ("Live Webcam Predict", "Capture Image And Predict",":rainbow[Multiple Images Upload -]🖼️🖼️🖼️", "Upload Video"),
							captions = ["Live Count in Zone. :red[(Slow)]🐌", "Click and Detect. :orange[(Recommended)] :green[(Super Fast)]⚡⚡", "Upload & Process Multiple Images. :orange[(Recommended)] :green[(Fast)]⚡", "Upload Video & Predict 🏗️:orange[(Work in Progress)]📽️🎞️"], index = 1)
	conf = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)
	if choice == "Live Webcam Predict":
		# conf = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

		# Define the WebRTC client settings
		client_settings = ClientSettings(
			# rtc_configuration={
			# 	"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
			# },
			media_stream_constraints={
				"video": True,
				"audio": False,
			},
		)

		# Define the WebRTC video transformer
		class ObjectDetector(VideoTransformerBase):

			def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
				# Convert the frame to an image
				# img = Image.fromarray(frame.to_ndarray())
				img = frame.to_ndarray(format="bgr24").copy()
				# Flip the image horizontally
				img = np.flip(img, axis=1)
				results = model.predict(img)[0]

				if isinstance(results, list):
					results1 = results[0]  
				else:
					results1 = results

				annotated_frame = results1.plot()
				annotated_frame = av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
				return annotated_frame

		# Start the WebRTC streamer
		webrtc_streamer(
			key="object-detection",
			mode=WebRtcMode.SENDRECV,
			# rtc_configuration={
			# 	"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
			# },
			media_stream_constraints={
				"video": True,
				"audio": False,
			},
			video_processor_factory=ObjectDetector,)
	elif choice == "Capture Image And Predict":
		img_file_buffer = st.camera_input("Take a picture")

		if img_file_buffer is not None:
			# To read image file buffer with OpenCV:
			bytes_data = img_file_buffer.getvalue()
			cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

			results = model.predict(cv2_img)
			# st.write(results)
			if isinstance(results, list):
				results1 = results[0]  
			else:
				results1 = results

			annotated_frame = results1.plot()
			# st.image(annotated_frame.to_ndarray(), channels="BGR")
			st.image(annotated_frame, channels="BGR")
			try:
				classifications = sv.Classifications.from_ultralytics(results1)

				# Get the top class ID and confidence
				top_class_ids, top_confidences = classifications.get_top_k(1)
				top_class_label = f"Class: {top_class_ids[0]}, Confidence: {top_confidences[0]:.2f}"
				# cv2.putText(cv2_img, top_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
				# Display the annotated image on Streamlit
				# st.image(cv2_img, channels="BGR")


				st.write(':orange[ Info : ⤵️ ]')
				output_dict = {}
				output_dict[top_class_label] = []

				for r in results:
				    cls_r = r.probs.data
				    values = cls_r.tolist()
				    cls_name = model.names
				    cls_lst = list(cls_name.values())
				    class_info = {"Class Name": cls_lst, "Confidence": values}
				    output_dict[top_class_label].append(class_info)
				# Convert the dictionary to JSON format
				output_json = json.dumps(output_dict, indent=4)

				# Print or use the JSON output as needed
				st.json(output_json)










				# st.write(':orange[ Info : ⤵️ ]')
				# for r in results:
				# 	st.write(f':green[{top_class_label}]')
				# 	# st.write(dir(r))
				# 	cls_r = r.probs.data
				# 	# probs = st.write(cls_r)
				# 	values = cls_r.tolist()
				# 	# st.write(values)
				# 	print("values: ->",values)

				# 	cls_name = model.names
				# 	cls_lst = list(cls_name.values())
				# 	# class_names = st.write(cls_lst)
				# 	print("class_names: ->",cls_lst)
				# 	hh = f"Class Name: {cls_lst}, Confidence: {values}"

				# 	st.write(hh)



			except AttributeError as e:
				for r in results:
					st.json(r.tojson())
				pass
			st.subheader("",divider='rainbow')

	elif choice == ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️":
		uploaded_files = st.file_uploader("Choose a images", type=['png', 'jpg', 'webp', 'bmp'], accept_multiple_files=True)
		for uploaded_file in uploaded_files:
			bytes_data = uploaded_file.read()
			st.write("filename:", uploaded_file.name)
			bytes_data = uploaded_file.getvalue()
			cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

			results = model.predict(cv2_img)

			if isinstance(results, list):
				results1 = results[0]  
			else:
				results1 = results

			annotated_frame = results1.plot()
			# st.image(annotated_frame.to_ndarray(), channels="BGR")
			st.image(annotated_frame, channels="BGR")
			try:
				classifications = sv.Classifications.from_ultralytics(results1)

				# Get the top class ID and confidence
				top_class_ids, top_confidences = classifications.get_top_k(1)
				top_class_label = f"Class: {top_class_ids[0]}, Confidence: {top_confidences[0]:.2f}"
				# cv2.putText(cv2_img, top_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
				# Display the annotated image on Streamlit
				# st.image(cv2_img, channels="BGR")


				st.write(':orange[ Info : ⤵️ ]')
				output_dict = {}
				output_dict[top_class_label] = []

				for r in results:
				    cls_r = r.probs.data
				    values = cls_r.tolist()
				    cls_name = model.names
				    cls_lst = list(cls_name.values())
				    class_info = {"Class Name": cls_lst, "Confidence": values}
				    output_dict[top_class_label].append(class_info)
				# Convert the dictionary to JSON format
				output_json = json.dumps(output_dict, indent=4)

				# Print or use the JSON output as needed
				st.json(output_json)










				# st.write(':orange[ Info : ⤵️ ]')
				# for r in results:
				# 	st.write(f':green[{top_class_label}]')
				# 	# st.write(dir(r))
				# 	cls_r = r.probs.data
				# 	# probs = st.write(cls_r)
				# 	values = cls_r.tolist()
				# 	# st.write(values)
				# 	print("values: ->",values)

				# 	cls_name = model.names
				# 	cls_lst = list(cls_name.values())
				# 	# class_names = st.write(cls_lst)
				# 	print("class_names: ->",cls_lst)
				# 	hh = f"Class Name: {cls_lst}, Confidence: {values}"

				# 	st.write(hh)



			except AttributeError as e:
				for r in results:
					st.json(r.tojson())
				pass
			# st.json(labels1)
			st.subheader("",divider='rainbow')
	elif choice == "Upload Video":
		st.title("🏗️Work in Progress📽️🎞️")
		'''# Gaurang is Working on it...
		clip = st.file_uploader("Choose a video file", type=['mp4'])

		if clip:
					# Read the content of the video file
			video_content = clip.read()
			# Convert the video content to a bytes buffer
			video_buffer = BytesIO(video_content)
			st.video(video_buffer)
			with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
				temp_filename = temp_file.name
				temp_file.write(video_content)

				print(f"---->_{temp_filename}")
				results = model(temp_filename,show = False, stream=True, save = False)
				for r in results:
					boxes = r.boxes  # Boxes object for bbox outputs
					masks = r.masks  # Masks object for segment masks outputs
					probs = r.probs  # Class probabilities for classification outputs
					orig_img = r.orig_img
					video_path = temp_filename  # Replace with the path to your input video

					cap = cv2.VideoCapture(video_path)
					fourcc = cv2.VideoWriter_fourcc(*'mp4v')
					with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file_o:
						temp_filename1 = temp_file_o.name
						output_path = temp_filename1  # Replace with the desired output path
						out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (int(cap.get(3)), int(cap.get(4))))
						# Assuming `results` is a generator, convert it to a list
						results_list = list(results)
						for frame_number in range(len(results_list)):  # Get total number of frames
							ret, frame = cap.read()
							
							# Get detection results for the current frame
							results_for_frame = results_list[frame_number]
							boxes = results_for_frame.boxes.xyxy.cpu().numpy()  # Assuming xyxy format
							masks = results_for_frame.masks.tensor.cpu().numpy() if results_for_frame.masks is not None else None
							# Check if probabilities are available
							if results_for_frame.probs is not None:
								# Get class names based on class indices
								class_names_dict = results_for_frame.names
								class_indices = results_for_frame.probs.argmax(dim=1).cpu().numpy()
								class_names = [class_names_dict[class_idx] for class_idx in class_indices]
							else:
								class_names = []

							# Draw annotations on the frame
							annotated_frame = draw_annotations(frame.copy(), boxes, masks, class_names)

							# Save the annotated frame to the output video
							out.write(annotated_frame)
							
						cap.release()
						out.release()
						print(f"___{output_path}")

						# output video
						import base64
						# Display the annotated video
						video_bytes = open(output_path, "rb")
						video_buffer2 = video_bytes.read()
						st.video(video_buffer2)
						st.success("Video processing completed.")




				# # video_content1 = results.read()
				# # Convert the video content to a bytes buffer
				# video_buffer1 = BytesIO(results)

			

			# st.success("Video processing completed.")'''



				# # Read the content of the video file
				# # video_content1 = results.read()
				# # Convert the video content to a bytes buffer
				# video_buffer1 = BytesIO(results)
				# st.video(video_buffer1)

			# Display the processed video
			# st.video(output_path)            
			# st.video(results)
			

			# st.success("Video processing completed.")

	
	st.subheader("",divider='rainbow')
	st.write(':orange[ Classes : ⤵️ ]')
	cls_name = model.names
	cls_lst = list(cls_name.values())
	st.write(f':orange[{cls_lst}]')
if __name__ == '__main__':
	main()
