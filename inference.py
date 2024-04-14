import tensorflow as tf
import cv2
import numpy as np
from glob import glob
import gradio as gr

### Moodify Fuction ###
def Moodify(img, quality=0.8):
    # grayscale img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(gray.shape)
    # get face bounding box coordinates using Haar Cascade
    faces = face_haar_cascade.detectMultiScale(gray, 1.3, 5)
    song = ' '
    play = ''
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        predictions = loaded_model.predict(img_pixels)
        emotion_label = np.argmax(predictions)

        emotion_prediction = label_dict[emotion_label]

        cv2.putText(img, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1 )

        resize_image = cv2.resize(img, (1000,700))
        cv2_imshow(resize_image)
        print("\n\n Emotion - ", emotion_prediction)
        if(emotion_prediction == 'Happiness' or emotion_prediction == 'Surprise'):
            play = random.choice(happy_links)
            song = happy[happy_links.index(play)]
        elif(emotion_prediction == 'Disgust' or emotion_prediction == 'Angry'):
            play = random.choice(energetic_links)
            song = energetic[energetic_links.index(play)]
        elif(emotion_prediction == 'Neutral'):
            play = random.choice(calm_links)
            song = calm[calm_links.index(play)]
        elif(emotion_prediction == 'Sad' or emotion_prediction == 'Fear'):
            play = random.choice(sad_links)
            song = sad[sad_links.index(play)]
    print("\n\n Now Playing - ", song)
    return play, emotion_prediction

def RunMoodify(img):
    player, emotion = Moodify(img)
    print("Moodify done")
    # vid = YouTubeVideo(player, width=1, height=1, allow_autoplay = True)
    vid = f"""<iframe width=100% height=490 max-width:100% src='https://www.youtube.com/embed/{player}?autoplay=1' title='Moodify Player' frameborder='0' allow='autoplay'></iframe>"""
    return gr.HTML(vid), emotion

if __name__=="__main__":
    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Moodify #")
        with gr.Row():
            with gr.Column():
                img = gr.Image(source="webcam", streaming=True)
                btn = gr.Button(value="Capture")
            with gr.Column():
                output_html = gr.HTML("")
                emotion = gr.Label("")
        btn.click(RunMoodify, inputs=[img], outputs=[output_html, emotion])

    demo.launch(debug=True)