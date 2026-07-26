import os  # Για να διαβάζουμε φακέλους και αρχεία από τα Windows
import cv2  # Η OpenCV: Τη θέλουμε για να ανοίγουμε και να αλλάζουμε μέγεθος (resize) στις εικόνες
import numpy as np  # Βιβλιοθήκη για μαθηματικά και πίνακες (οι εικόνες είναι πίνακες αριθμών)
import matplotlib.pyplot as plt  # Για να σχεδιάζουμε διαγράμματα
import seaborn as sns  # Για να κάνουμε τον πίνακα σύγχυσης (Confusion Matrix) πιο εμφανίσιμο
from sklearn.model_selection import train_test_split  # Έτοιμη συνάρτηση για τον χωρισμό 70-30
from sklearn.metrics import confusion_matrix, classification_report  # Για τον υπολογισμό της ακρίβειας
import tensorflow as tf  # Η βασική βιβλιοθήκη Deep Learning
from tensorflow.keras.models import Sequential  # Το "κουτί" που θα βάλουμε μέσα τα επίπεδα του δικτύου
# Εισαγωγή των επιπέδων (Layers) που θα χτίσουν το AlexNet
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam  # Ο αλγόριθμος που διορθώνει τα βάρη (Optimizer)

# ΡΥΘΜΙΣΕΙΣ (CONSTANTS)
DATA_DIR = "101_ObjectCategories"  # Το όνομα του φακέλου με τις εικόνες
IMG_SIZE = (224, 224)  # Ορίζουμε ότι ολες οι εικόνες θα γίνουν 224x224 pixels (απαραίτητο για CNN)
BATCH_SIZE = 32        # Το μοντέλο θα βλέπει τις εικόνες σε πακέτα των 32 (όχι μία-μία, ούτε όλες μαζί)
EPOCHS = 20            # Πόσες φορές θα περάσει το μοντέλο πάνω από ΟΛΑ τα δεδομένα για να μάθει
NUM_CLASSES_TO_USE = 101 # Πόσες κατηγορίες θα χρησιμοποιήσουμε (101 για το αρχικό ερώτημα)

#  Φόρτωση Δεδομένω
def load_data(data_dir, img_size, max_classes=None):
    images = []  # pixels των εικόνων
    labels = []  # σε ποια κατηγορία ανήκουν ( 0=αεροπλάνο, 1=πρόσωπο)
    
    # τα ονόματα των φακέλων (είναι οι κατηγορίες μας)
    class_names = sorted(os.listdir(data_dir))
    
    # Αν θέλουμε λιγότερες κλάσεις (50 για το ερώτημα 7), κόβουμε τη λίστα
    if max_classes:
        class_names = class_names[:max_classes]
        
    # Φτιάχνουμε ένα λεξικό για να αντιστοιχίσουμε ονόματα σε αριθμούς (π.χ. 'airplanes': 0)
    class_map = {name: i for i, name in enumerate(class_names)}
    
    print(f"Φόρτωση {len(class_names)} κλάσεων...")
    
    # μπαίνουμε σε κάθε φάκελο κατηγορίας
    for category in class_names:
        path = os.path.join(data_dir, category) # Η διαδρομή του φακέλου 
        class_num = class_map[category]         # Ο αριθμός της κατηγορίας
        
        # Αν δεν είναι φάκελος αλλά αρχείο, το αγνοούμε
        if not os.path.isdir(path): continue
        
        # Μπαίνουμε μέσα στον φάκελο και διαβάζουμε κάθε εικόνα
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name) # Η πλήρης διαδρομή της εικόνας
                img_array = cv2.imread(img_path)        # Διαβάζουμε την εικόνα με OpenCV
                
                if img_array is None: continue          # Αν η εικόνα είναι χαλασμένη, την προσπερνάμε
                
                #Αλλάζουμε το μέγεθος σε 224x224 γιατί το CNN θέλει σταθερή είσοδο
                img_array = cv2.resize(img_array, img_size)
                
                images.append(img_array) # Αποθηκεύουμε την εικόνα
                labels.append(class_num) # Αποθηκεύουμε την ετικέτα της
            except Exception as e:
                pass # Αν γίνει κάποιο λάθος συνεχίζει στην επόμενη
    
    # Μετατρέπουμε τις λίστες σε πίνακες Numpy (πιο γρήγορο)
    # Normalization βοηθάει το νευρωνικό να μάθει πιο γρήγορα.
    X = np.array(images).astype('float32') / 255.0
    y = np.array(labels)
    
    return X, y, class_names

#Διαχωρισμός (Splitting)
def split_data(X, y, method='ratio'):
    # Ερώτημα 3α: Κλασικός διαχωρισμός 70% Train - 30% Test
    if method == 'ratio':
        # Το stratify=y φροντίζει να κρατήσει τις αναλογίες των κλάσεων ίδιες
        return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Ερώτημα 3β: Ίσος αριθμός δειγμάτων ανά κλάση
    elif method == 'equal':
        TARGET_TRAIN_SAMPLES = 30 # Θέλουμε 30 εικόνες ανά κλάση για εκπαίδευση
        
        X_train, X_test, y_train, y_test = [], [], [], []
        classes = np.unique(y) # Βρίσκουμε ποιες είναι οι κλάσεις (0, 1, 2...)
        
        for c in classes:
            # Βρίσκουμε τις θέσεις (indices) των εικόνων που ανήκουν στην κλάση c
            idx = np.where(y == c)[0]
            np.random.shuffle(idx) # Ανακατεύουμε τυχαία για να μην πάρουμε μόνο τις πρώτες
            total_images = len(idx)
            
            # Εδώ είναι η διόρθωση για τις μικρές κλάσεις:
            # Αν μια κλάση έχει λίγες εικόνες, δεν παίρνουμε 30, αλλά το 80% 
            # ώστε να μείνει οπωσδήποτε κάτι για το Test set.
            if total_images <= TARGET_TRAIN_SAMPLES + 2:
                split_point = int(total_images * 0.8)
            else:
                split_point = TARGET_TRAIN_SAMPLES
            
            # Διόρθωση ασφαλείας για να μην βγούμε εκτός ορίων
            if split_point >= total_images:
                split_point = total_images - 1
                
            # Χωρίζουμε τα indices σε train και test
            train_idx = idx[:split_point]
            test_idx = idx[split_point:]
            
            # Προσθέτουμε τις εικόνες στις λίστες
            X_train.extend(X[train_idx])
            y_train.extend(y[train_idx])
            X_test.extend(X[test_idx])
            y_test.extend(y[test_idx])
            
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# Κατασκευή Μοντέλου (AlexNet) ---
def build_alexnet(num_classes, img_size=(224, 224, 3)):
    model = Sequential() # Ξεκινάμε ένα κενό μοντέλο
    
    
    # Conv2D: 96 φίλτρα ψάχνουν μοτίβα.
    # Kernel (11,11): Το μέγεθος του φακού που σαρώνει την εικόνα.
    # Strides (4,4): Το βήμα σάρωσης (ανά 4 pixels).
    model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='valid', input_shape=img_size))
    model.add(Activation('relu')) # ReLU: Κόβει τις αρνητικές τιμές (κάνει το δίκτυο μη γραμμικό)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2))) # Μειώνει το μέγεθος της εικόνας (κρατάει τα σημαντικά)
    model.add(BatchNormalization()) # Κανονικοποιεί τα δεδομένα ΜΕΣΑ στο δίκτυο (βοηθάει να μη "κολλάει" η εκπαίδευση)

    # Περισσότερα φίλτρα για να μάθουμε πιο σύνθετα μοτίβα
    model.add(Conv2D(256, (5, 5), padding='same')) # Padding='same': Κρατάει τις διαστάσεις ίδιες
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Βαθύτερα χαρακτηριστικά
    model.add(Conv2D(384, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # CLASSIFIER
    model.add(Flatten()) # Μετατρέπει τον 3D πίνακα εικόνας σε μία μακριά γραμμή (διάνυσμα)

    # Πρώτο πυκνό επίπεδο (Fully Connected) με 4096 νευρώνες
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Dropout: "Σβήνει" τυχαία το 50% των νευρώνων σε κάθε βήμα εκπαίδευσης.
    # Σκοπός: Να μην αποστηθίσει (overfit) το μοντέλο τα δεδομένα.
    model.add(Dropout(0.5)) 

    # Δεύτερο πυκνό επίπεδο
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # ΤΕΛΙΚΟ ΕΠΙΠΕΔΟ: Έχει τόσους νευρώνες όσες και οι κλάσεις μας (π.χ. 101)
    model.add(Dense(num_classes))
    # Softmax: Μετατρέπει τους αριθμούς εξόδου σε πιθανότητες (π.χ. 80% αεροπλάνο, 2% γάτα...)
    model.add(Activation('softmax'))

    # Ρύθμιση εκπαίδευσης
    model.compile(optimizer=Adam(learning_rate=0.0001), # Adam: Ο "εξυπνος" αλγόριθμος που αλλάζει τα βάρη
                  loss='sparse_categorical_crossentropy', # Η συνάρτηση που μετράει το λάθος
                  metrics=['accuracy']) # Θέλουμε να παρακολουθούμε την ακρίβεια
    return model

#ΚΥΡΙΩΣ ΠΡΟΓΡΑΜΜΑ
print("Φόρτωση δεδομένων...")
# Έλεγχος αν υπάρχει ο φάκελος
if not os.path.exists(DATA_DIR):
    print(f"ΣΦΑΛΜΑ: Δεν βρέθηκε ο φάκελος {DATA_DIR}.")
else:
    # 1. Καλούμε τη συνάρτηση φόρτωσης
    X, y, class_names = load_data(DATA_DIR, IMG_SIZE, max_classes=NUM_CLASSES_TO_USE)

    # 2. Καλούμε τη συνάρτηση διαχωρισμού (εδώ επιλέγουμε 'ratio' ή 'equal')
    print("Διαχωρισμός δεδομένων...")
    X_train, X_test, y_train, y_test = split_data(X, y, method='ratio') 

    # 3. Φτιάχνουμε το μοντέλο
    print("Δημιουργία και εκπαίδευση μοντέλου AlexNet...")
    model = build_alexnet(len(class_names))
    
    # 4. Ξεκινάει η εκπαίδευση (Εδώ περιμένεις ώρα!)
    history = model.fit(X_train, y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=(X_test, y_test)) # Σε κάθε εποχή ελέγχει την ακρίβεια στο Test set

    # 5. Τελική Αξιολόγηση
    print("Αξιολόγηση...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%") # Τυπώνει το ποσοστό επιτυχίας

    # Κάνουμε προβλέψεις για να φτιάξουμε τον πίνακα
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1) # Παίρνουμε την κλάση με τη μεγαλύτερη πιθανότητα

    # Βρίσκουμε ποιες κλάσεις υπάρχουν ΟΝΤΩΣ μέσα στο y_test
    unique_labels = np.unique(y_test)
    # Κρατάμε μόνο τα ονόματα που αντιστοιχούν σε αυτές τις κλάσεις
    valid_target_names = [class_names[i] for i in unique_labels]

    # Εκτύπωση Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title(f'AlexNet Confusion Matrix (Acc: {acc:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\nClassification Report:")
    # Περνάμε τη λίστα ονομάτων
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=valid_target_names))