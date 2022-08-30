import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import time
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model

# importing the tflite Movenet model
MoveNetmodel = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
MoveNetmodel.allocate_tensors()

# Importing the Yoga Classification model
model2 = load_model('SinglePOseYogaDetector.h5')

# Setting demo paths if none is given by user
DEMO_IMAGE = 'Demos/demo_image.jpg'
DEMO_VIDEO = 'Demos/demo_video.mp4'

# Specifying the class names to get the prediction
actions = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']


# Detection and drawing Functions
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


def draw_wrong_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


# Initiating the streamlit
st.title('Yoga Pose Detection App')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div: first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div: first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Yoga Pose Sidebar')
st.sidebar.subheader('parameters')


# noinspection PyTypeChecker,PyUnusedLocal
@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['About App', 'Detect on Image', 'Detect on Video']
                                )
if app_mode == 'About App':
    st.markdown('In this application we are using **Movenet model** to detect the 5 classes of yoga poses. ')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div: first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div: first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.image('display/Display_Image.png')

    st.markdown(
        '''
        # About Me \n
          Hey this is **Basim Bashir** from **Invictus Solutions**. \n
          
          I am a **Machine Learning Enthusiast** and **Data Scientist**. \n
          
          Check me out on Social Media: \n
          - [LinkedIn](https://www.linkedin.com/in/basim-bashir-035403214/) \n
          - [Gmail](mailto:basim.bashir0968@gmail.com) 
        ''')


elif app_mode == 'Detect on Image':
    st.sidebar.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div: first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div: first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Detected Pose**")
    kpi1_text = st.markdown("No Class")

    # Set the number of detection i.e., single or multi person
    threshold = st.sidebar.number_input("Classification Threshold", value=0.5, min_value=0.0, max_value=1.0)
    st.sidebar.markdown('---')

    # Set the Detection confidence threshold
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    # Defining a Browse file Dialogue Box functionality
    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        # noinspection PyTypeChecker
        image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        # noinspection PyTypeChecker
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    # ------------------------------------------------------#
    # DashBoard

    # Copying image to show on output
    out_image = image.copy()

    # saving the image to get the proper dimensions
    cv2.imwrite('out_image.jpg', out_image)
    # reading the image from the root directory again to get proper alignments
    frame = cv2.imread('out_image.jpg')
    # resizing the image to have 192*192*3 (MoveNet specified Tensor)
    resized = cv2.resize(frame, (192, 192), interpolation=cv2.INTER_LINEAR)

    # Expanding the dimensions
    img = tf.image.resize_with_pad(np.expand_dims(resized, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = MoveNetmodel.get_input_details()
    output_details = MoveNetmodel.get_output_details()

    # Make predictions
    MoveNetmodel.set_tensor(input_details[0]['index'], np.array(input_image))
    MoveNetmodel.invoke()
    keypoints_with_scores = MoveNetmodel.get_tensor(output_details[0]['index'])

    # Flattening the Keypoints
    flat = keypoints_with_scores.flatten()

    # Reshaping to 51 columns
    reshaped = flat.reshape(-1, 1)

    # Predicting the results
    res = model2.predict(np.expand_dims(reshaped, axis=0))[0]

    # Checking if the detected Keypoints are same as the classes then print Class name
    if res[np.argmax(res)] > threshold:
        # Make a h1 tag header
        kpi1_text.write(f"<h1 style='text-align: center; color:red;'> {actions[np.argmax(res)]} </h1>", unsafe_allow_html=True)

        # Rendering
        draw_connections(resized, keypoints_with_scores, EDGES, detection_confidence)
        draw_keypoints(resized, keypoints_with_scores, detection_confidence)
    else:
        # Rendering
        draw_wrong_connections(resized, keypoints_with_scores, EDGES, detection_confidence)
        draw_keypoints(resized, keypoints_with_scores, detection_confidence)

    # Display the Image
    st.subheader('Output Image')
    st.image(resized, use_column_width=True)


elif app_mode == 'Detect on Video':

    # To suppress any deprecation warning on the app
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # To give the option to the End User to use webcam or not
    use_webcam = st.sidebar.button('Use Webcam')
    # make a checkbox to record the video
    record = st.sidebar.checkbox("Record Video")

    # If record button is pressed then start recording
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div: first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div: first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Set the threshold for prediction i.e., it displays the class on basis of this threshold
    threshold = st.sidebar.number_input("Classification Threshold", value=0.5, min_value=0.0, max_value=1.0)
    st.sidebar.markdown('---')

    # Set the Detection confidence threshold
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    # display the output
    st.markdown("## Output")

    # create an empty frame
    stframe = st.empty()

    # Get the Video from user
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4,' 'mov', 'avi', 'asf', 'm4v'])

    # Make it a Temporary video file in the root directory
    tffile = tempfile.NamedTemporaryFile(delete=False)

    # If we don't have a video file then instantiate webcam or default video to display
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)

        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    # Get the video or Live cam video's width and height of the frame
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the FPS for the video
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # Recording part
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    # creating the Front-End for the video part
    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    # Initializing fps and iterations to 0 from start
    fps = 0
    i = 0

    # ------------------------------------------------------#
    # Pose Classification Logic

    # Creating columns to display fps and detections etc
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Pose**")
        kpi2_text = st.markdown("No Class")

    with kpi3:
        st.markdown("**Video Resolution**")
        kpi3_text = st.markdown("0")

    with kpi4:
        st.markdown("**Mis-Classified Ratio**")
        kpi4_text = st.markdown("0")

    # Creating a divider line
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Now Detection Loop part
    prevTime = 0

    while vid.isOpened():
        i += 1
        ret, frame = vid.read()

        if not ret:
            break

        # Reshape image
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        # Setup input and output
        input_details = MoveNetmodel.get_input_details()
        output_details = MoveNetmodel.get_output_details()

        # Make predictions
        MoveNetmodel.set_tensor(input_details[0]['index'], np.array(input_image))
        MoveNetmodel.invoke()
        keypoints_with_scores = MoveNetmodel.get_tensor(output_details[0]['index'])

        # Flattening the Keypoints
        flat = keypoints_with_scores.flatten()

        # Reshaping to 51 columns
        reshaped = flat.reshape(-1, 1)

        # Predicting the results
        res = model2.predict(np.expand_dims(reshaped, axis=0))[0]

        # Checking if the detected Keypoints are same as the classes then print Class name
        if res[np.argmax(res)] > threshold:
            # Make a h1 tag header
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'> {actions[np.argmax(res)]} </h1>",
                            unsafe_allow_html=True)

            # Rendering
            draw_connections(frame, keypoints_with_scores, EDGES, detection_confidence)
            draw_keypoints(frame, keypoints_with_scores, detection_confidence)

            # Mis-Classification difference
            diff = 100 - (threshold * 100)
            diff = round(diff)
            kpi4_text.write(f"<h1 style='text-align: center; color:red;'> {diff} </h1>",
                            unsafe_allow_html=True)

        else:
            # Rendering
            draw_wrong_connections(frame, keypoints_with_scores, EDGES, detection_confidence)
            draw_keypoints(frame, keypoints_with_scores, detection_confidence)

        # Fps counter logic
        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        # putting fps into the recording
        if record:
            out.write(frame)

        # ------------------------------------------------------#
        # DashBoard
        kpi1_text.write(f"<h1 style='text-align: center; color:red;'> {int(fps)} </h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color:red;'> {width, height} </h1>", unsafe_allow_html=True)

        # Resizing the frame
        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='BGR', use_column_width=True)
