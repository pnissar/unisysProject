# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import network
# import guided_filter
# from tqdm import tqdm
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()



# def resize_crop(image):
#     h, w, c = np.shape(image)
#     if min(h, w) > 720:
#         if h > w:
#             h, w = int(720 * h / w), 720
#         else:
#             h, w = 720, int(720 * w / h)
#     image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
#     h, w = (h // 8) * 8, (w // 8) * 8
#     image = image[:h, :w, :]
#     return image


# def cartoonize(load_folder, save_folder, model_path):
#     tf.disable_v2_behavior()

#     input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
#     network_out = network.unet_generator(input_photo)
#     final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

#     all_vars = tf.trainable_variables()
#     gene_vars = [var for var in all_vars if 'generator' in var.name]
#     saver = tf.train.Saver(var_list=gene_vars)

#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)

#     # Convert to absolute path and fix slashes
#     model_path = os.path.abspath(model_path)
#     print(f"Looking for checkpoints in: {model_path}")

#     checkpoint_path = tf.train.latest_checkpoint(model_path)
#     if checkpoint_path is None:
#         raise ValueError(f"No checkpoint found in {model_path}. Check if the model files exist.")

#     print(f"Restoring model from {checkpoint_path}")
#     saver.restore(sess, checkpoint_path)

#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     name_list = os.listdir(load_folder)
#     for name in tqdm(name_list):
#         try:
#             load_path = os.path.join(load_folder, name)
#             save_path = os.path.join(save_folder, name)

#             image = cv2.imread(load_path)
#             if image is None:
#                 print(f"Failed to load image: {load_path}")
#                 continue

#             image = resize_crop(image)
#             batch_image = image.astype(np.float32) / 127.5 - 1
#             batch_image = np.expand_dims(batch_image, axis=0)

#             output = sess.run(final_out, feed_dict={input_photo: batch_image})
#             output = (np.squeeze(output) + 1) * 127.5
#             output = np.clip(output, 0, 255).astype(np.uint8)

#             cv2.imwrite(save_path, output)

#         except Exception as e:
#             print(f"Error processing {load_path}: {e}")


# if __name__ == '__main__':
#     script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory

#     model_path = os.path.join(script_dir, "saved_models")
#     load_folder = os.path.join(script_dir, "test_images")
#     save_folder = os.path.join(script_dir, "cartoonized_images")

#     if not os.path.exists(load_folder):
#         raise ValueError(f"Load folder does not exist: {load_folder}")

#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     cartoonize(load_folder, save_folder, model_path)










import os
import cv2
import numpy as np
import tensorflow as tf
import network
import guided_filter
import streamlit as st
from PIL import Image
import tempfile

tf.compat.v1.disable_v2_behavior()

# Load TensorFlow model
def load_model(model_path):
    input_photo = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)
    
    all_vars = tf.compat.v1.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.compat.v1.train.Saver(var_list=gene_vars)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {model_path}. Check if model files exist.")
    
    saver.restore(sess, checkpoint_path)
    return sess, input_photo, final_out

# Image preprocessing
def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    return image[:h, :w, :]

# Cartoonize function
def cartoonize_image(image, sess, input_photo, final_out):
    image = resize_crop(image)
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

# Ensure cartoonized_images directory exists
cartoonized_dir = "cartoonized_images"
os.makedirs(cartoonized_dir, exist_ok=True)

# Streamlit UI
st.title("Cartoonizer App 🖌️🎨")
st.write("Upload an image to convert it into a cartoon!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    model_path = os.path.join(os.path.dirname(__file__), "saved_models")
    sess, input_photo, final_out = load_model(model_path)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cartoonized_image = cartoonize_image(image, sess, input_photo, final_out)
    
    st.image(cartoonized_image, caption="Cartoonized Image", use_column_width=True)
    
    # Save the cartoonized image in cartoonized_images directory
    cartoonized_path = os.path.join(cartoonized_dir, "cartoonized_output.jpg")
    cv2.imwrite(cartoonized_path, cv2.cvtColor(cartoonized_image, cv2.COLOR_RGB2BGR))
    
    with open(cartoonized_path, "rb") as file:
        st.download_button(label="Download Cartoonized Image", data=file, file_name="cartoonized.jpg", mime="image/jpeg")
