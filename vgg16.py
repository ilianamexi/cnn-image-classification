import os   # Για διαχείριση φακέλων (ανάγνωση ονομάτων αρχείων)
import cv2  # OpenCV: Για φόρτωση και αλλαγή μεγέθους εικόνων
import numpy as np # Για μαθηματικές πράξεις και πίνακες
import matplotlib.pyplot as plt # Για τα διαγράμματα
import seaborn as sns # Για την ωραία απεικόνιση του Confusion Matrix
# Συνάρτηση για τον διαχωρισμό Train/Test
from sklearn.model_selection import train_test_split 
# Συναρτήσεις για μέτρηση απόδοσης (Πίνακας Σύγχυσης, Ακρίβεια)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf # Η κύρια βιβλιοθήκη Deep Learning
# Φορτώνουμε το έτοιμο μοντέλο VGG16 από το Keras
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Model # Για να φτιάξουμε το τελικό μας μοντέλο
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input # Τα επίπεδα που θα προσθέσουμε
from tensorflow.keras.optimizers import Adam # Ο αλγόριθμος εκπαίδευσης
from tensorflow.keras.utils import to_categorical

#ΡΥΘΜΙΣΕΙΣ (CONSTANTS)
DATA_DIR = "101_ObjectCategories"  # Ο φάκελος με τις εικόνες
IMG_SIZE = (224, 224) # Το VGG16 εκπαιδεύτηκε σε αυτές τις διαστάσεις, οπότε πρέπει να τις τηρήσουμε
BATCH_SIZE = 32       # Επεξεργασία 32 εικόνων τη φορά
EPOCHS = 10           # Transfer Learning
NUM_CLASSES_TO_USE = 101 # Πόσες κατηγορίες θα κρατήσουμε

#Φόρτωση Δεδομένων
def load_data(data_dir, img_size, max_classes=None):
    """Φορτώνει τις εικόνες και τις ετικέτες."""
    images = [] # Λίστα για τα pixels
    labels = [] # Λίστα για τους αριθμούς κατηγοριών
    class_names = sorted(os.listdir(data_dir)) # Διαβάζουμε τα ονόματα των φακέλων
    
    # Αν θέλουμε μόνο 50 κλάσεις (για το ερώτημα 7), κόβουμε τη λίστα εδώ
    if max_classes:
        class_names = class_names[:max_classes]
        
    # Αντιστοίχιση ονόματος σε αριθμό 
    class_map = {name: i for i, name in enumerate(class_names)}
    
    print(f"Φόρτωση {len(class_names)} κλάσεων...")
    
    for category in class_names:
        path = os.path.join(data_dir, category)
        class_num = class_map[category]
        if not os.path.isdir(path): continue # Αν δεν είναι φάκελος, τον αγνοούμε
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path) # Διάβασμα εικόνας
                if img_array is None: continue   # Έλεγχος αν η εικόνα είναι κατεστραμμένη
                
                # Resize στις διαστάσεις που θέλει το VGG16 (224x224)
                img_array = cv2.resize(img_array, img_size)
                
                images.append(img_array)
                labels.append(class_num)
            except Exception as e:
                pass
    
    # Μετατροπή σε numpy array και διαίρεση με 255 για να πάνε οι τιμές στο [0, 1]
    X = np.array(images).astype('float32') / 255.0 
    y = np.array(labels)
    return X, y, class_names

#Διαχωρισμός Δεδομένων 
def split_data(X, y, method='ratio'):
    """
    Διαχωρίζει τα δεδομένα.
    method: 'ratio' -> 70% train, 30% test (Ερώτημα 3α)
    method: 'equal' -> Σταθερό πλήθος για train (Ερώτημα 3β)
    """
    if method == 'ratio':
        # stratify=y: Κρατάει την αναλογία των κλάσεων ίδια με το αρχικό dataset
        return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    elif method == 'equal':
        # Στόχος: Να πάρουμε 30 εικόνες από κάθε κατηγορία για εκπαίδευση
        TARGET_TRAIN_SAMPLES = 30 
        
        X_train, X_test, y_train, y_test = [], [], [], []
        classes = np.unique(y) # Βρίσκουμε όλες τις μοναδικές κατηγορίες
        
        for c in classes:
            idx = np.where(y == c)[0] # Βρίσκουμε τις θέσεις των εικόνων αυτής της κατηγορίας
            np.random.shuffle(idx)    # Τις ανακατεύουμε
            total_images = len(idx)
            
            
            # Αν μια κλάση έχει π.χ. μόνο 31 εικόνες, δεν παίρνουμε 30 γιατί θα μείνει μόνο 1 για test.
            # Παίρνουμε το 80% για να είμαστε σίγουροι.
            if total_images <= TARGET_TRAIN_SAMPLES + 2:
                split_point = int(total_images * 0.8) 
            else:
                split_point = TARGET_TRAIN_SAMPLES
            
            # Ασφάλεια για να μην βγούμε εκτός ορίων
            if split_point >= total_images:
                split_point = total_images - 1
                
            train_idx = idx[:split_point] # Οι πρώτες πάνε για Train
            test_idx = idx[split_point:]  # Οι υπόλοιπες για Test
            
            X_train.extend(X[train_idx])
            y_train.extend(y[train_idx])
            X_test.extend(X[test_idx])
            y_test.extend(y[test_idx])
            
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

#Κατασκευή Μοντέλου (VGG16) 
def build_model(num_classes):
    """Φτιάχνει το μοντέλο VGG16 με Transfer Learning."""
    
    # 1. Φόρτωση του VGG16
    # weights='imagenet': Φέρνει τα "έξυπνα" βάρη που έμαθε το μοντέλο βλέποντας εκατομμύρια εικόνες.
    # include_top=False: ΠΕΤΑΜΕ το τελευταίο κομμάτι του VGG (τον ταξινομητή των 1000 κλάσεων).
    # Θα βάλουμε τον δικό μας ταξινομητή για τις 101 κλάσεις.
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # 2. Πάγωμα Βαρών (Freezing)
    # Λέμε στο δίκτυο: "Μην πειράξεις αυτά τα επίπεδα κατά την εκπαίδευση".
    # Κρατάμε τη γνώση που έχει ήδη (πώς να βλέπει γραμμές, σχήματα, μάτια, ρόδες κτλ).
    for layer in base_model.layers:
        layer.trainable = False
        
    # 3. Προσθήκη Δικού μας Ταξινομητή (Custom Classifier)
    # Flatten: Παίρνει τα τετραγωνάκια (features) που έβγαλε το VGG και τα κάνει μια ευθεία γραμμή.
    x = Flatten()(base_model.output)
    
    # Ένα πυκνό επίπεδο 256 νευρώνων για να μάθει τους συνδυασμούς των χαρακτηριστικών.
    x = Dense(256, activation='relu')(x)
    
    # Dropout 0.5: Σβήνει τυχαία τους μισούς νευρώνες για να αποφύγουμε την αποστήθιση (overfitting).
    x = Dropout(0.5)(x)
    
    # Το τελικό επίπεδο απόφασης. Έχει τόσους νευρώνες όσες οι κλάσεις (101).
    # Softmax: Βγάζει πιθανότητες.
    output = Dense(num_classes, activation='softmax')(x)
    
    # Ενώνουμε το παλιό (base) με το καινούργιο (output) κομμάτι
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile: Ρυθμίζουμε πώς θα μαθαίνει.
    # Χρησιμοποιούμε μικρό Learning Rate (0.0001) για να κάνουμε λεπτομερείς διορθώσεις.
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# MAIN
# 1. Φόρτωση
print("Φόρτωση δεδομένων...")
X, y, class_names = load_data(DATA_DIR, IMG_SIZE, max_classes=NUM_CLASSES_TO_USE)

# 2. Διαχωρισμός
# ΕΔΩ ΕΙΝΑΙ Η ΑΛΛΑΓΗ: Έβαλες method='equal', άρα κάνεις το Ερώτημα 3β!
print("Διαχωρισμός δεδομένων (Μέθοδος Ίσου Πλήθους - 3β)...")
X_train, X_test, y_train, y_test = split_data(X, y, method='equal') 

# 3. Εκπαίδευση
print("Δημιουργία και εκπαίδευση μοντέλου VGG16...")
model = build_model(len(class_names))

# Ξεκινάει η εκπαίδευση. 
# validation_data=(X_test, y_test): Σε κάθε εποχή ελέγχει πόσο καλά πάει στις άγνωστες εικόνες.
history = model.fit(X_train, y_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    validation_data=(X_test, y_test))

# 4. Αξιολόγηση
print("Αξιολόγηση...")
# Υπολογίζει την τελική ακρίβεια στο Test set
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Το μοντέλο κάνει προβλέψεις για όλες τις εικόνες του Test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1) # Παίρνουμε την κλάση με τη μέγιστη πιθανότητα

# 5. Σχεδίαση Πίνακα Σύγχυσης (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(20, 20)) # Μεγάλο μέγεθος εικόνας για να φαίνονται οι 101 κλάσεις
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title(f'VGG16 Confusion Matrix (Acc: {acc:.2f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 6. Αναλυτική Αναφορά (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))