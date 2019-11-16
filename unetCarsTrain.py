from model import *
from data import *
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import os
import time
from BackgroundParser import SourcePreparation

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_flag = True
tune_flag = False
is_export = False

image_width = 1024
image_height = 576


n_epochs = 30
batch_size = 1
train_sample_len = 5184
test_sample_len = 500
#test_sample_len = 2
train_steps = train_sample_len//batch_size
test_steps = test_sample_len//batch_size
learning_rate = 1e-4

current_dir = os.getcwd()
project_dir = os.path.dirname(current_dir)

# train_path = current_dir+'/dataset/augmented_dataset/train'
# val_path = current_dir+'/dataset/augmented_dataset/test'
# test_path = current_dir+'/dataset/augmented_dataset/test/color'
#
# model_path = current_dir+'/models/unet2019-11-02-09-22-38.21-tloss-0.1277-tdice-0.8723-vdice-0.7350.hdf5' #512

train_path = current_dir+'/dataset/augmented_dataset/train'
val_path = current_dir+'/dataset/augmented_dataset/test'
result_path = current_dir+"/dataset/results_multiclass_unetMCT_1"


test_path = current_dir+"/dataset/augmented_dataset/train/source"

model_path = current_dir+'/models/unet2019-11-16-01-29-22.19-tloss-0.7558-tdice-0.2442-vdice-0.2046.hdf5'

timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
log_filename = current_dir+'/logs/log-unet-'+ timestr +'.txt'
with open(log_filename, "a") as log_file:
    log_file.write("Experiment start: "+ log_filename + "\n")
    log_file.write("Input dataset(s):\n")
    log_file.write("Train path: "+ train_path +"\n")
    log_file.write("Val path: "+ val_path +"\n")

def read_and_preprocess_image(test_path,filename,target_size = (256,256)):

    raw = cv2.imread(os.path.join(test_path, filename))
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = raw / 255
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = np.reshape(img, (1,) + img.shape)
    scale_coef = (raw.shape[0]/target_size[0], raw.shape[1]/target_size[1])

    return img, raw, scale_coef
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
data_gen_args = dict()
with open(log_filename, "a") as log_file:
    log_file.write("Augmentation args: "+ str(data_gen_args) + "\n")

# trainGene = trainGenerator(batch_size=batch_size,
#                         train_path=train_path,
#                         image_folder='color',
#                         mask_folder='mask',
#                         aug_dict=data_gen_args,
#                         image_color_mode='rgb',
#                         target_size = (image_height, image_width),
#                         save_to_dir = None) # project_dir+'/data/dataset/aug')
# valGene = trainGenerator(batch_size=1,
#                         train_path=val_path,
#                         image_folder='color',
#                         mask_folder='mask',
#                         aug_dict=dict(),
#                         image_color_mode='rgb',
#                         target_size = (image_height, image_width),
#                         save_to_dir = None)
# testGene = testGenerator(test_path=test_path,
#                          num_image = test_sample_len,
#                          target_size = (image_height, image_width))

source_preparator = SourcePreparation()
label_list = source_preparator.label_list
color_mask_dict = source_preparator.color_mask_dict
num_class = len(label_list)

trainGene = trainGenerator(batch_size=batch_size,
                        train_path=train_path,
                        image_folder='source',
                        mask_folder='mask',
                        aug_dict=data_gen_args,
                        image_color_mode='rgb',
                        mask_color_mode = "rgb",
                        flag_multi_class = True,
                        num_class=num_class,
                        target_size = (image_height, image_width),
                        save_to_dir = None,
                        label_list=label_list,
                        color_mask_dict=color_mask_dict) # project_dir+'/data/dataset/aug')
valGene = trainGenerator(batch_size=1,
                        train_path=val_path,
                        image_folder='source',
                        mask_folder='mask',
                        aug_dict=dict(),
                        image_color_mode='rgb',
                        mask_color_mode = "rgb",
                        flag_multi_class = True,
                        num_class=num_class,
                        target_size = (image_height, image_width),
                        save_to_dir = None,
                        label_list=label_list,
                        color_mask_dict=color_mask_dict
                        )
testGene = testGenerator(test_path=test_path,
                         num_image = test_sample_len,
                         target_size = (image_height, image_width))
if not is_export:
    if not tune_flag:
        if train_flag:
            model_path = None

with open(log_filename, "a") as log_file:
    log_file.write("Model path: "+ str(model_path) + "\n")



if is_export:
    '''import tensorflow as tf
    from tensorflow.tools.graph_transforms import TransformGraph
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    from keras import backend as K

    # Get keras model and save
    model_keras = model.keras_model
    # All new operations will be in test mode from now on.
    K.set_learning_phase(0)
    sess = K.get_session()'''



    import tensorflow as tf
    from tensorflow.python.framework import graph_io
    from tensorflow.keras.models import load_model

    #tf.keras.backend.clear_session()

    save_pb_dir = 'models'
    model_fname = "models/keras_model_UnetMCT_softmax.h5"


    def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
        with graph.as_default():
            graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
            graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
            return graphdef_frozen


    # This line must be executed before loading Keras model.
    tf.keras.backend.set_learning_phase(0)

    # model = unet_light_ct_tv(pretrained_weights=model_path, input_size=(image_height, image_width, 3),
    #     #                    learning_rate=learning_rate, n_classes=num_class, no_compile = True)
    model = unet_less_light_ct(pretrained_weights=model_path, input_size=(image_height, image_width, 3),
                        learning_rate=learning_rate, n_classes=num_class, no_compile = True)

    model.save(model_fname)
    model = load_model(model_fname)


    session = tf.keras.backend.get_session()
    #init = tf.global_variables_initializer()
    #session.run(init)


    input_names = [t.op.name for t in model.inputs]
    output_names = [t.op.name for t in model.outputs]

    # Prints input and output nodes names, take notes of them.
    print(input_names, output_names)

    frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs],
                                save_pb_dir=save_pb_dir, save_pb_name='frozen_model_ct_unetmct_softmax.pb')

    #import tensorflow.contrib.tensorrt as trt

    # from tensorflow.python.compiler.tensorrt import trt_convert as trt
    #
    # trt_graph = trt.create_inference_graph(
    #     input_graph_def=frozen_graph,
    #     outputs=output_names,
    #     max_batch_size=1,
    #     max_workspace_size_bytes=1 << 25,
    #     precision_mode='FP16',
    #     minimum_segment_size=3
    # )
    #
    # graph_io.write_graph(trt_graph, "models/",
    #                      "unet_trt_graph.pb", as_text=False)

else:
    #model = unet_light_ct_tv(pretrained_weights=model_path, input_size=(image_height, image_width, 3),
    # learning_rate=learning_rate, n_classes=num_class)
    # model = unet_less_light_ct(pretrained_weights=model_path, input_size=(image_height, image_width, 3),
    #                                  learning_rate = learning_rate, n_classes = num_class)
    model = unet(pretrained_weights=model_path, input_size=(image_height, image_width, 3),
                 learning_rate = learning_rate, n_classes = num_class)

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)



    with open(log_filename, "a") as log_file:
        log_file.write('\nModel summary:\n')
        log_file.write(short_model_summary)
        log_file.write('\nOptimizer:' +str(model.optimizer)+' Learning rate: '+str(learning_rate)+'\n')
        log_file.write('Loss:' + str(model.loss) + '\n')
        log_file.write('Metrics:' + str(model.metrics) + '\n')
        log_file.write('Image_Size: ' + str(image_width) + ' x '+str(image_height) + '\n')
        log_file.write('Batch_Size: ' + str(batch_size) + '\n')
        log_file.write('\nTraining process:\n')
        log_file.write('\nEpoch no, train dice, train loss, val dice, val loss\n')
    if train_flag:
        if num_class == 1:
            model_checkpoint = ModelCheckpoint(current_dir+'/models/unet'+timestr+
                                               '.{epoch:02d}-tloss-{loss:.4f}-tdice-{dice_coef:.4f}-vdice-{val_dice_coef:.4f}.hdf5',
                                               monitor='loss', verbose=1, save_best_only=True, save_weights_only = True)
        else:
            model_checkpoint = ModelCheckpoint(current_dir + '/models/unet' + timestr +
                                               '.{epoch:02d}-tloss-{loss:.4f}-tdice-{dice_coef_multilabel:.4f}-vdice-{val_dice_coef_multilabel:.4f}.hdf5',
                                               monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
        callbacks = [model_checkpoint, callbacks.CSVLogger(log_filename, separator=',', append=True)]

        start_time = time.time()

        model.fit_generator(trainGene,steps_per_epoch=train_steps,epochs=n_epochs,validation_data=valGene, validation_steps=test_sample_len, callbacks=callbacks)

        end_time = time.time()
        duration = end_time - start_time
        with open(log_filename, "a") as log_file:
            log_file.write("Training time, sec: "+ str(duration) + "\n")
    else:
        os.makedirs(result_path, exist_ok=True)

        for filename in sorted(os.listdir(test_path)):
            img, raw, scale_coef = read_and_preprocess_image(test_path=test_path,
                                                             filename=filename,
                                                             target_size=(image_height, image_width))
            start_time = time.time()
            results = model.predict(img, verbose=1)
            end_time = time.time()
            duration = end_time - start_time
            # with open(log_filename, "a") as log_file:
            #     log_file.write("Testing time, sec: " + str(duration) + "\n")

            out_file_name = os.path.join(result_path, filename)
            img = labelVisualize(label_list, color_mask_dict, results[0])
            io.imsave(out_file_name, img)

            # start_time = time.time()
            # results = model.predict_generator(testGene,test_sample_len,verbose=1)
            # end_time = time.time()
            # duration = end_time - start_time
            # with open(log_filename, "a") as log_file:
            #     log_file.write("Testing time, sec: " + str(duration) + "\n")
            #
            # if num_class == 1:
            #     os.makedirs(current_dir + "/dataset/results", exist_ok=True)
            #     saveResult(current_dir+"/dataset/results",results)
            # else:
            #     os.makedirs(result_path, exist_ok=True)
            #     saveResult(result_path, results, flag_multi_class=True,
            #                label_list=label_list, color_mask_dict=color_mask_dict)
