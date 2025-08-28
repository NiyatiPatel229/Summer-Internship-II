# --- Task 1: Get all image paths ---
import glob
all_image_paths = glob.glob('./images_dataSAT/class_0_non_agri/*') + \
                  glob.glob('./images_dataSAT/class_1_agri/*')
print(len(all_image_paths))

# --- Task 2: Create temp by binding image paths and labels, print 5 random samples ---
import random
labels = [0]*len(glob.glob('./images_dataSAT/class_0_non_agri/*')) + \
         [1]*len(glob.glob('./images_dataSAT/class_1_agri/*'))
temp = list(zip(all_image_paths, labels))
print(random.sample(temp, 5))

# --- Task 3: Generate batch using custom_data_generator ---
def custom_data_generator(image_paths, labels, batch_size):
    while True:
        idxs = np.random.choice(len(image_paths), batch_size)
        batch_X = [np.array(Image.open(image_paths[i])) for i in idxs]
        batch_y = [labels[i] for i in idxs]
        yield np.stack(batch_X), np.array(batch_y)

gen = custom_data_generator(all_image_paths, labels, batch_size=8)
X_batch, y_batch = next(gen)
print(X_batch.shape, y_batch.shape)

# --- Task 4: Create validation data batch size = 8 ---
# Example using keras.utils.Sequence or a generator as above
val_gen = custom_data_generator(all_image_paths, labels, batch_size=8)
val_X, val_y = next(val_gen)

