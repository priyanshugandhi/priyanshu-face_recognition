import os
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *


tf.app.flags.DEFINE_string('checkpoint', 'checkpoint', 'Checkpoint basename')

FLAGS = tf.app.flags.FLAGS

def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file):
            try:
                
                image_batch = make_multi_crop_batch(image_file, coder)
                
                batch_results = sess.run(softmax_output, feed_dict = {images:image_batch})
                        
                output = batch_results[0]
                batch_sz = batch_results.shape[0]
            
                for i in range(1, batch_sz):
                    output = output + batch_results[i]

                output /= batch_sz
                best = np.argmax(output)
                best_choice = (label_list[best], output[best]*100)
                        
                return best_choice
            
                nlabels = len(label_list)
                if nlabels > 2:
                    output[best] = 0
                    second_best = np.argmax(output)
                    second_best_choice=(label_list[second_best],output[second_best]*100)
                    return second_best_choice
                        
            except Exception as e:
                print(e)
                print('Failed to run image %s ' % image_file)

def model_init(sess, model_path, label_list): 
            
            model_checkpoint_path, global_step = get_checkpoint(model_path, None, FLAGS.checkpoint)

            nlabels = len(label_list)
            model_fn = select_model(FLAGS.model_type)
            images_placeholder = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            
            with tf.device(FLAGS.device_id):
                    
                logits = model_fn(nlabels, images_placeholder, 1, False)
                    
                init = tf.global_variables_initializer()
                                   
                saver = tf.train.Saver()
                saver.restore(sess, model_checkpoint_path)
                                    
                softmax_output = tf.nn.softmax(logits)
                                
                return softmax_output, images_placeholder  

def main(filename,sess,sess1,AGE_LIST,GENDER_LIST,age_softmax,age_image,gender_softmax,gender_image):

            files = []
   
            files.append(filename)
                
            
            coder_age = ImageCoder()
            coder_gender = ImageCoder()

            image_files = list(filter(lambda x: x is not None, [files[0]]))
            
            for image_file in image_files:
                best_data,second_best_data=classify_one_multi_crop(sess, AGE_LIST, age_softmax, coder_age, age_image, image_file)
                gender_data=classify_one_multi_crop(sess1, GENDER_LIST, gender_softmax, coder_gender, gender_image, image_file)
            
            age_gender_data=[]
            age_gender_data.append(best_data)
            age_gender_data.append(int(second_best_data))
            age_gender_data.append(gender_data[0])
            return age_gender_data
            sess.close()
            sess1.close()
