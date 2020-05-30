import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

"""
Wir benutzen das Modul models von keras, um ein neues Neuronales Netz zu erstellen.
Der Sequential Konstruktor erstellt dieses für uns. 
"""
model = tf.keras.models.Sequential()

"""
Nun haben wir also unser Modell, 
welches jedoch noch keine Layer hat. Diese müssen wir erst hinzufügen.
"""
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

"""
Um einen Layer hinzuzufügen, benutzen wir die Funktion add unseres Modells. Dann können wir von dem Modul layers einen Layer-Typen auswählen. 
Für den Input-Layer werden wir einen Flatten Layer, mit der Eingangsform 28x28 auswählen. 
Im Klartext bedeutet das, dass wir 784 Neuronen in unserem Input-Layer haben (Produkt aus 28 mal 28). Flatten gibt hierbei nur an, dass wir 784 Neuronen in einer Reihe haben anstatt in einem 28x28 Gitter. 
Diese Eingangsneuronen stellen die Pixel unserer Beispiele da. 
Unser Ziel ist es, dass wir durch diese Pixel am Ende auf eine Lösung kommen (also eine Klassifikation).
"""

model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

"""
Als nächstes fügen wir zwei Dense Layer hinzu. 
Diese stellen unsere Hidden-Layer da und sollen die Komplexität unseres Modells erhöhen. 
Beide Layer haben jeweils 128 Neuronen. 
Als Aktivierungsfunktion haben wir hier, bei dem Parameter activation, die ReLU-Funktion gewählt (siehe Kapitel 1). 
Dense bedeutet hierbei nur, dass jedes Neuron dieser Schicht, mit jedem Neuron der letzten bzw. nächsten Schicht verbunden ist. 
Also ein ganz „normaler“ Layer.
"""
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

"""
Zu guter Letzt definieren wir nun noch unseren Output-Layer. 
Auch dieser ist hierbei ein Dense-Layer, hat jedoch nur zehn Neuronen und eine andere Aktivierungsfunktion. 
Die zehn Neuronen geben am Ende an, wie sehr unser Modell glaubt, dass es sich bei den eingegebenen Pixeln, um die jeweilige Zahl handelt. 
Das erste Neuron steht für die Null, das zweite für die Eins etc. Die Aktivierungsfunktion, welche wir hier gewählt haben ist eine ganz besondere – die Softmax Funktion. 
Diese sorgt dafür, dass die Ergebnisse des Output-Layers eine Summe von 1 ergeben. 
Sie wandelt also die absoluten Werte in relative Ergebnisse um, sodass wir sehen zu wie viel Prozent eine Ziffer vorhergesagt wird. model = tf.keras.models.Sequential()
"""
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

"""
Bevor wir nun mit dem Trainieren und Testen anfangen, müssen wir dieses Modell kompilieren. 
Dabei optimieren wir es und definieren ebenso die Loss Function.
"""
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""
Nun geht es an den Hauptteil des Ganzen – das Trainieren und das Testen. Hierzu benötigen wir jeweils nur eine Zeile.
Mit der fit Funktion trainieren wir unser Modell. 
Hierzu übergeben wir die x- und y-Trainingswerte.
Als zusätzlichen Parameter geben wir die Anzahl der Epochen an.
Diese Zahl gibt an, wie oft unser Modell dieselben Beispiele sehen wird.
"""
model.fit(X_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)
