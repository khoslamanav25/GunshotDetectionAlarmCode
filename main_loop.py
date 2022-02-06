from convnet_model import other_datagen, model, train_generator
import numpy as np

directory = 'mic_recorded_sounds/spectrograms'

new_image_generator = other_datagen.flow_from_directory(
        directory=directory,
        target_size=(64, 64),
        batch_size=64,
        shuffle=False
)



try:
    new_image_generator.reset()
    pred = model.predict_generator(new_image_generator, steps=100, verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)
    #labels from train gen for testing
    
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    print(predictions)
    
except:
    print('invalid')
        

